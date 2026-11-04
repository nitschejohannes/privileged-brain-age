#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Helpers
# -----------------------------
def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name in {"silu", "swish"}:
        return nn.SiLU(inplace=True)
    if name == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    raise ValueError(f"Unknown activation: {name}")

def build_mlp(
    in_dim: int,
    hidden: Tuple[int, ...] = (256, 128),
    out_dim: int = 128,
    activation: str = "gelu",
    dropout: float = 0.2,
    layer_norm: bool = True,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim

    for h in hidden:
        layers.append(nn.Linear(prev, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(_activation(activation))
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h

    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)

# -----------------------------
# Config
# -----------------------------
@dataclass
class GDModelConfig:
    activation: str = "gelu"
    dropout: float = 0.2
    layer_norm: bool = True
    rep_hidden: Tuple[int, ...] = (256, 128)
    rep_dim: int = 128
    head_hidden: Tuple[int, ...] = (128,)

# -----------------------------
# Age distribution helpers
# -----------------------------
def make_age_grid(
    y_train: Union[torch.Tensor, list, tuple],
    *,
    age_bin_step: float = 1.0,
    margin_bins: int = 1,
) -> torch.Tensor:
    """
    Build a 1D age grid from the training ages only.
    """
    y = torch.as_tensor(y_train, dtype=torch.float32).view(-1)
    y_min = torch.floor(y.min()) - margin_bins * age_bin_step
    y_max = torch.ceil(y.max()) + margin_bins * age_bin_step

    num = int(torch.round((y_max - y_min) / age_bin_step).item()) + 1
    grid = y_min + torch.arange(num, dtype=torch.float32) * age_bin_step
    return grid

def gaussian_soft_labels(
    y: torch.Tensor,
    age_grid: torch.Tensor,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Encode chronological age as a Gaussian-smoothed probability vector over age_grid.
    Shape:
        y         : (B,) or (B,1)
        age_grid  : (K,)
        returns   : (B,K)
    """
    if sigma <= 0:
        raise ValueError("sigma must be > 0.")

    y = y.view(-1, 1).float()
    grid = age_grid.view(1, -1).float().to(y.device)
    probs = torch.exp(-0.5 * ((grid - y) / sigma) ** 2)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return probs

def logits_to_probs(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0.")
    return F.softmax(logits / temperature, dim=1)

def probs_to_expected_age(probs: torch.Tensor, age_grid: torch.Tensor) -> torch.Tensor:
    grid = age_grid.view(1, -1).to(probs.device)
    return (probs * grid).sum(dim=1)

def logits_to_expected_age(
    logits: torch.Tensor,
    age_grid: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    probs = logits_to_probs(logits, temperature=temperature)
    return probs_to_expected_age(probs, age_grid)

# -----------------------------
# Losses
# -----------------------------
def soft_cross_entropy_from_logits(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Cross-entropy with a soft target distribution.
    Equivalent to KL(target || pred) up to the entropy of the target.
    """
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(target_probs * log_probs).sum(dim=1)

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss

def rkd_distance_loss(
    student_values: torch.Tensor,
    teacher_values: torch.Tensor,
    *,
    detach_teacher: bool = True,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Relational Knowledge Distillation - distance term (Park et al., 2019).

    Computes pairwise distances between samples in the batch separately for
    student and teacher, normalizes each set of distances by its mean
    (scale invariance), and penalizes the smooth-L1 discrepancy between them.

    This preserves the *relative geometry* of the teacher's predictions across
    subjects, rather than matching absolute values or full distributions.

    Parameters
    ----------
    student_values, teacher_values : torch.Tensor
        Tensors of shape (B,) or (B, D). Typically BAGs (B,) or expected ages
        (B,) for this brain-age setting; can also be representations (B, D).
    detach_teacher : bool
        Whether to stop gradients from flowing into the teacher branch.
    eps : float
        Stabilizer for the mean-distance normalization.
    reduction : {'mean', 'sum', 'none'}
        Reduction over the pairwise terms.
    Returns
    -------
    torch.Tensor
        Scalar loss (or vector if reduction='none').
    Notes
    -----
    Requires batch size B >= 2; with B == 1 there are no pairs and the loss
    is returned as 0 (with grad). The pairwise tensor has C(B,2) entries.
    """
    if student_values.shape != teacher_values.shape:
        raise ValueError(
            f"student_values and teacher_values must have the same shape; "
            f"got {tuple(student_values.shape)} vs {tuple(teacher_values.shape)}."
        )

    if detach_teacher:
        teacher_values = teacher_values.detach()

    # Promote (B,) -> (B, 1) so torch.pdist works uniformly.
    if student_values.dim() == 1:
        student_values = student_values.unsqueeze(1)
        teacher_values = teacher_values.unsqueeze(1)

    B = student_values.shape[0]
    if B < 2:
        # No pairs; return a 0 tensor that still tracks grad on the student branch.
        return student_values.sum() * 0.0

    s_d = torch.pdist(student_values, p=2)
    t_d = torch.pdist(teacher_values, p=2)

    # Scale-invariant normalization by the mean pairwise distance.
    s_d = s_d / s_d.mean().clamp_min(eps)
    t_d = t_d / t_d.mean().clamp_min(eps)

    return F.smooth_l1_loss(s_d, t_d, reduction=reduction)

def combined_student_classification_loss(
    target_probs: torch.Tensor,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    age_grid: torch.Tensor,
    y: torch.Tensor,
    *,
    lambda_kd: float = 0.5,
    kd_method: str = "rkd",
    use_residual_bag: bool = True,
    rkd_scale: float = 50.0,
    distill_temperature: float = 2.0,
    return_parts: bool = False,
):
    """
    Generalized distillation objective with selectable KD method:

    kd_method="rkd" (default, backward-compatible):
        (1 - λ) * CE(y_soft, s_logits)
        + λ     * rkd_scale * RKD_dist(student_BAGs, teacher_BAGs)

    kd_method="soft_label" (Hinton et al., 2015):
        (1 - λ) * CE(y_soft, s_logits)
        + λ     * T² * CE(softmax(t_logits/T), s_logits/T)

    Method selection guidance
    -------------------------
    Use "rkd" when teacher and student share a representational basis (e.g.
    both use brain-imaging features from different modalities like sMRI and
    fMRI). The student's feature space can express the teacher's pairwise
    BAG geometry, so the relational constraint is compatible with supervised
    learning.

    Use "soft_label" when the modality gap is large (e.g. sMRI teacher,
    questionnaire student). In this setting, RKD's relational constraint
    forces the student to reorganize its internal geometry to match a
    structure its features cannot naturally express, producing systematically
    negative gradient cosines between the supervised and KD objectives.
    Soft-label KD avoids this by only requiring the student to shift its
    scalar predicted-age distribution toward the teacher's — an operation
    the student can achieve by reweighting its existing features through the
    output head, without restructuring its representation.

    Parameters
    ----------
    target_probs : (B, K)
        Gaussian-smoothed soft labels from chronological age.
    student_logits : (B, K)
        Student's raw logits over the age grid.
    teacher_logits : (B, K)
        Teacher's raw logits over the age grid.
    age_grid : (K,)
        The shared age grid.
    y : (B,)
        Chronological ages (used by RKD for BAG computation).
    lambda_kd : float
        Convex mixing weight for the KD term. 0 = pure supervised.
    kd_method : {"rkd", "soft_label"}
        Which KD loss to use.
    use_residual_bag : bool
        [RKD only] If True, match BAG geometry; if False, match expected-age
        geometry. Ignored when kd_method="soft_label".
    rkd_scale : float
        [RKD only] Multiplier that brings RKD loss (~1e-2) into the same
        range as supervised CE (~1e0). Ignored when kd_method="soft_label"
        (both terms are cross-entropies over the same grid, so they are
        already on a comparable scale; T² handles the temperature correction).
    distill_temperature : float
        [soft_label only] Temperature for softening teacher and student
        distributions. T=1.0 uses the teacher's hard predictions; T=2-4
        spreads probability mass to neighboring age bins, exposing "dark
        knowledge" about which ages the teacher considers plausible. Start
        with T=2.0 for a ~60-bin age grid with sigma=1.0 soft labels.
        Ignored when kd_method="rkd".
    return_parts : bool
        If True, return (total, l_sup, l_kd_scaled).

    Returns
    -------
    total loss, or (total, l_sup, l_kd_scaled) if return_parts.
    l_kd_scaled includes rkd_scale (for RKD) or T² (for soft_label) so that
    weighted_kd = lambda_kd * l_kd_scaled matches the contribution to total.
    """
    if not (0.0 <= lambda_kd <= 1.0):
        raise ValueError("lambda_kd must be in [0, 1].")
    if kd_method not in ("rkd", "soft_label"):
        raise ValueError(f"kd_method must be 'rkd' or 'soft_label', got '{kd_method}'.")

    # ── Supervised term: identical for both methods ────────────────────
    l_sup = soft_cross_entropy_from_logits(student_logits, target_probs, reduction="mean")

    # ── λ = 0: no KD contribution (baseline-equivalent guarantee) ─────
    if lambda_kd == 0.0:
        l_kd_scaled = student_logits.detach().sum() * 0.0
        total = l_sup
        if return_parts:
            return total, l_sup, l_kd_scaled
        return total

    # ── KD term: method-dependent ─────────────────────────────────────
    if kd_method == "rkd":
        if rkd_scale <= 0.0:
            raise ValueError("rkd_scale must be > 0.")

        s_age = logits_to_expected_age(student_logits, age_grid, temperature=1.0)
        t_age = logits_to_expected_age(teacher_logits, age_grid, temperature=1.0)

        if use_residual_bag:
            y_flat = y.view(-1).to(s_age.device).float()
            s_val = s_age - y_flat
            t_val = t_age - y_flat
        else:
            s_val = s_age
            t_val = t_age

        l_kd_raw = rkd_distance_loss(
            student_values=s_val,
            teacher_values=t_val,
            detach_teacher=True,
            reduction="mean",
        )
        l_kd_scaled = rkd_scale * l_kd_raw

    elif kd_method == "soft_label":
        if distill_temperature <= 0.0:
            raise ValueError("distill_temperature must be > 0.")
        # Teacher's softened distribution (detached — no grad into teacher).
        teacher_probs_T = logits_to_probs(
            teacher_logits.detach(), temperature=distill_temperature,
        )
        # Student's CE against the teacher's soft targets, at the same
        # temperature. soft_cross_entropy_from_logits applies log_softmax
        # internally, so we pass student_logits / T.
        l_kd_raw = soft_cross_entropy_from_logits(
            student_logits / distill_temperature,
            teacher_probs_T,
            reduction="mean",
        )
        # T² scaling (Hinton et al., 2015): the gradient of CE w.r.t.
        # logits/T is 1/T times the gradient w.r.t. logits.  Multiplying
        # by T² compensates so that (a) gradient magnitudes are stable
        # across temperature choices and (b) the KD loss remains on the
        # same scale as the supervised CE at T=1.
        l_kd_scaled = (distill_temperature ** 2) * l_kd_raw

    # ── Combined objective ────────────────────────────────────────────
    total = (1.0 - lambda_kd) * l_sup + lambda_kd * l_kd_scaled

    if return_parts:
        return total, l_sup, l_kd_scaled
    return total

# -----------------------------
# Models
# -----------------------------
class TeacherClassifier(nn.Module):
    """
    Teacher classifier over age bins.
    Input: privileged features.
    Output: class logits over age_grid.
    """
    def __init__(self, x_dim: int, n_classes: int, cfg: GDModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_classes = n_classes

        self.rep = build_mlp(
            in_dim=x_dim,
            hidden=cfg.rep_hidden,
            out_dim=cfg.rep_dim,
            activation=cfg.activation,
            dropout=cfg.dropout,
            layer_norm=cfg.layer_norm,
        )
        self.logit_head = build_mlp(
            in_dim=cfg.rep_dim,
            hidden=cfg.head_hidden,
            out_dim=n_classes,
            activation=cfg.activation,
            dropout=cfg.dropout,
            layer_norm=cfg.layer_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.rep(x)
        return self.logit_head(h)

class StudentClassifier(nn.Module):
    """
    Student / baseline classifier over age bins.
    Input: regular features.
    Output: class logits over age_grid.
    """

    def __init__(self, x_dim: int, n_classes: int, cfg: GDModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_classes = n_classes

        self.rep = build_mlp(
            in_dim=x_dim,
            hidden=cfg.rep_hidden,
            out_dim=cfg.rep_dim,
            activation=cfg.activation,
            dropout=cfg.dropout,
            layer_norm=cfg.layer_norm,
        )
        self.logit_head = build_mlp(
            in_dim=cfg.rep_dim,
            hidden=cfg.head_hidden,
            out_dim=n_classes,
            activation=cfg.activation,
            dropout=cfg.dropout,
            layer_norm=cfg.layer_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.rep(x)
        return self.logit_head(h)

@torch.no_grad()
def teacher_predict_probs(
    teacher: TeacherClassifier,
    x: torch.Tensor,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = teacher(x)
    probs = logits_to_probs(logits, temperature=temperature)
    return logits, probs