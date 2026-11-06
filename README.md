# Generalized Distillation for Normative Brain Age Modeling

Python code for evaluating generalized distillation in normative brain age modeling, using simulated data and UK Biobank structural MRI, resting-state fMRI and biosocial data. It accompanies the manuscript *Privilege Is Not Enough: Evaluating Generalized Distillation for Normative Brain Age Modeling in Depression, Anxiety and Alcohol Use Disorder*.

Folders are numbered in the order required to reproduce the results.

## Scripts

### [`00_simulation_based`](00_simulation_based)

| Script | Purpose |
|---|---|
| [`brain_age_sim.py`](00_simulation_based/brain_age_sim.py) | Generate the synthetic cohorts and define the simulation models. |
| [`run_brain_age_sim.py`](00_simulation_based/run_brain_age_sim.py) | Run the repetitions, then generate the simulation statistics and figures. |

### [`01_normative_model_development`](01_normative_model_development)

| Script | Purpose |
|---|---|
| [`teacher_and_student_models.py`](01_normative_model_development/teacher_and_student_models.py) | Define the teacher and student models and the distillation losses. |
| [`baseline.py`](01_normative_model_development/baseline.py) | Define the baseline model, which matches the student without the distillation term. |
| [`deconfounding.py`](01_normative_model_development/deconfounding.py) | Define the modality-specific linear deconfounding and its before-and-after comparison. |
| [`pipeline_deconfound.py`](01_normative_model_development/pipeline_deconfound.py) | Fit deconfounding on training folds and apply it to held-out data. |
| [`run_normative_model_tuning.py`](01_normative_model_development/run_normative_model_tuning.py) | Tune the teacher, student and baseline hyperparameters. |
| [`run_normative_modeling_pipeline.py`](01_normative_model_development/run_normative_modeling_pipeline.py) | Train the models and write predicted ages and brain age gaps (BAGs). |

### [`02_result_evaluation`](02_result_evaluation)

These scripts use the pipeline outputs and do not depend on each other.

| Script | Purpose |
|---|---|
| [`cross_model_raw_BAG_comparison.py`](02_result_evaluation/cross_model_raw_BAG_comparison.py) | Compare raw BAGs across models and test teacher-likeness. |
| [`BAG-severity_association_analysis.py`](02_result_evaluation/BAG-severity_association_analysis.py) | Test BAG associations with symptom severity and cognition. |
| [`plot_parallel_coordinates.py`](02_result_evaluation/plot_parallel_coordinates.py) | Compare student associations between imitation weights. |
| [`transfer_signature_plot.py`](02_result_evaluation/transfer_signature_plot.py) | Quantify signal recovery and the distillation-added BAG component. |

## Dependencies

Python 3.12

```bash
pip install numpy pandas scipy "scikit-learn>=1.4" torch matplotlib
```

## Data

UK Biobank data are not included. Access requires an application through the [UK Biobank portal](http://ukbiobank.ac.uk/register-apply).

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).