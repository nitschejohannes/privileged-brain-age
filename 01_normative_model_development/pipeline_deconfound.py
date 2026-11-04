#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from deconfounding import (
    BlockDeconfounder,
    split_smri_measures,
    is_qc,
    CONF_SEX,
    CONF_ETIV,
    CONF_MOTION,
    SITE_DUMMIES,
)

# -----------------------------------------------------------------------------
# Modality -> (feature-group, confound-set) partition
# -----------------------------------------------------------------------------
def build_modality_spec(
    modality: str,
    feature_cols: Sequence[str],
    *,
    site_cols: Sequence[str],
) -> List[Tuple[List[str], List[str]]]:
    """
    Return the groups argument for a BlockDeconfounder given a block's feature
    columns and which site dummy columns are actually present.

    modality in {"smri", "fmri", "questionnaire"}.
    """
    feature_cols = list(feature_cols)
    site = list(site_cols)

    if modality == "smri":
        volume_like, thickness = split_smri_measures(feature_cols)
        groups: List[Tuple[List[str], List[str]]] = []
        if volume_like:
            groups.append((volume_like, [CONF_ETIV, CONF_SEX] + site))
        if thickness:
            groups.append((thickness, [CONF_SEX] + site))
        return groups

    if modality == "fmri":
        return [(feature_cols, [CONF_MOTION, CONF_SEX] + site)]

    if modality == "questionnaire":
        # No residualization: one group with an empty confound set.
        return [(feature_cols, [])]

    raise ValueError(f"unknown modality '{modality}' (expected smri/fmri/questionnaire)")

# -----------------------------------------------------------------------------
# Fit-on-train / apply-forward deconfounder + scaler
# -----------------------------------------------------------------------------
class FoldDeconfounder:
    """
    Residualize (per modality spec) then z-score. Fit on training rows only;
    apply forward. Drop-in for the pipeline's per-fold preprocessing.

    Parameters
    ----------
    modality : "smri" | "fmri" | "questionnaire"
    feature_cols : the block's feature column names (order matches X columns)
    site_cols : site dummy columns present in the confound frame (may be empty)
    """

    def __init__(self, modality: str, feature_cols: Sequence[str], *, site_cols: Sequence[str]):
        self.modality = modality
        self.feature_cols = list(feature_cols)
        self.site_cols = list(site_cols)
        groups = build_modality_spec(modality, self.feature_cols, site_cols=self.site_cols)
        self._dec = BlockDeconfounder(self.feature_cols, groups)
        self._scaler: Optional[StandardScaler] = None

    def fit(self, X_train_raw: np.ndarray, conf_train: pd.DataFrame) -> "FoldDeconfounder":
        Xr = self._dec.fit(np.asarray(X_train_raw, np.float64), conf_train).transform(
            np.asarray(X_train_raw, np.float64), conf_train
        )
        self._scaler = StandardScaler().fit(Xr)
        return self

    def transform(self, X_raw: np.ndarray, conf: pd.DataFrame) -> np.ndarray:
        if self._scaler is None:
            raise RuntimeError("FoldDeconfounder.transform called before fit.")
        Xr = self._dec.transform(np.asarray(X_raw, np.float64), conf)
        return self._scaler.transform(Xr).astype(np.float32)

    def fit_transform(self, X_train_raw: np.ndarray, conf_train: pd.DataFrame) -> np.ndarray:
        return self.fit(X_train_raw, conf_train).transform(X_train_raw, conf_train)

# -----------------------------------------------------------------------------
# Column-role resolution for a merged frame
# -----------------------------------------------------------------------------
def resolve_confound_cols(df_cols: Sequence[str], site_prefix: str = "site_") -> Dict[str, Any]:
    """
    Identify the confound columns present in a merged dataframe and return the
    names needed to build confound frames and specs.
    """
    df_cols = list(df_cols)
    site_cols = [c for c in SITE_DUMMIES if c in df_cols]
    # also accept any other site_* columns beyond the canonical three
    site_cols += [c for c in df_cols if c.startswith(site_prefix) and c not in site_cols]
    present = {
        "sex": CONF_SEX in df_cols,
        "etiv": CONF_ETIV in df_cols,
        "motion": CONF_MOTION in df_cols,
        "site_cols": site_cols,
    }
    return present

def required_confounds_for(modality: str, site_cols: Sequence[str]) -> List[str]:
    """Confound column names a given modality's correction will read."""
    if modality == "smri":
        return [CONF_ETIV, CONF_SEX] + list(site_cols)
    if modality == "fmri":
        return [CONF_MOTION, CONF_SEX] + list(site_cols)
    if modality == "questionnaire":
        return []
    raise ValueError(f"unknown modality '{modality}'")

def split_features_and_qc(feature_cols: Sequence[str]) -> Tuple[List[str], List[str]]:
    """
    Partition a block's columns into (model features, quarantined QC columns).
    QC columns (rfMRI/T1 SNR, registration discrepancy, warping) are scan-quality
    proxies, not brain features -- kept out of the model, returned separately so
    they remain available for downstream sensitivity analyses.
    """
    qc = [c for c in feature_cols if is_qc(c)]
    feats = [c for c in feature_cols if c not in set(qc)]
    return feats, qc