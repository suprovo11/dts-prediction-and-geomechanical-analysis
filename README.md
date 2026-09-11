# Machine Learning-Based DTS Prediction and Geomechanical Analysis

## Overview

This repository contains a reproducible end-to-end machine learning workflow for predicting synthetic shear-sonic travel time (DTS) from well-log data and evaluating its application to geomechanical analysis.

The workflow is developed for Well 15/9-F-1 A in the Volve field and integrates data preprocessing, leak-free feature engineering, machine learning model benchmarking, out-of-sample prediction, and geomechanical calculations for fracture gradient and mud-weight window estimation.

The complete study can be executed through a single Python pipeline.

## Research Workflow

The pipeline consists of four primary stages:

1. Data preprocessing and leak-free feature engineering
2. Machine learning model training and evaluation
3. Geomechanical analysis using predicted DTS
4. Diagnostic figure generation

The default execution performs all stages sequentially.

```bash
python research_pipeline.py --input Research_Data.csv --outdir results
```

Individual stages can also be executed independently:

```bash
python research_pipeline.py --stage data
python research_pipeline.py --stage train
python research_pipeline.py --stage geomech
python research_pipeline.py --stage figures
```

## Methodology

The workflow uses measured well-log variables including DTC, GR, DEN, NEU, resistivity logs, and caliper data to develop predictive features for DTS.

Feature engineering includes logarithmic transformations of resistivity measurements and trailing rolling statistics for selected well-log variables. Target-derived variables such as measured Poisson's ratio, Vp, and Vs are excluded from the machine learning feature set to prevent target leakage.

Model performance is evaluated using depth-ordered validation rather than random train-test splitting. The data are divided into a shallow training interval and an unseen deeper holdout interval. Five-fold blocked cross-validation is performed within the training data.

## Machine Learning Models

The pipeline benchmarks machine learning models against empirical baselines.

Implemented models include:

* DTC baseline
* DTC and GR baseline
* Random Forest
* XGBoost, when available
* LightGBM, when available
* HistGradientBoosting as the fallback boosting model
* Multilayer Perceptron

Model performance is evaluated using RMSE, MAE, and R². The best-performing machine learning model is selected according to holdout RMSE.

## Geomechanical Analysis

Predicted DTS is subsequently propagated into an elastic-property and stress-analysis workflow.

The geomechanical stage estimates:

* Compressional and shear-wave velocities
* Vp/Vs ratio
* Poisson's ratio
* Shear modulus
* Young's modulus
* Overburden stress
* Pore pressure
* Minimum horizontal stress
* Fracture gradient
* Equivalent mud-weight window

The predicted DTS is evaluated on the deep holdout interval to maintain an out-of-sample assessment of its effect on the geomechanical calculations.

## Reproducibility and Data Leakage Control

Several design decisions are implemented to improve the reliability of the research workflow:

* Depth-ordered blocked validation is used instead of random splitting.
* Empirical DTC-based baselines provide reference models for evaluating machine learning improvement.
* Rolling features are calculated using trailing windows to maintain a causal structure.
* Target-derived variables are excluded from the predictive feature set.
* Geomechanical evaluation uses an unseen deep holdout interval.
* A fixed random state is used to improve reproducibility.

## Requirements

The core workflow requires:

* Python 3
* NumPy
* pandas
* scikit-learn
* SciPy
* Matplotlib
* joblib
* pyarrow

XGBoost and LightGBM are optional. If neither is available, the workflow automatically uses scikit-learn's HistGradientBoostingRegressor.

Install the required packages with:

```bash
pip install numpy pandas scikit-learn scipy matplotlib joblib pyarrow
```

Optional boosting libraries:

```bash
pip install xgboost lightgbm
```

## Input Data

The default input file is:

```text
Research_Data.csv
```

The pipeline expects the well-log data to contain the configured target and depth columns:

```text
DTS
MD
```

along with the primary predictive logs defined in the configuration.

The input structure, target variable, feature selection, validation parameters, and geomechanical assumptions can be modified through the configuration section of `research_pipeline.py`.

## Output

Running the complete pipeline generates a `results` directory containing the principal research outputs, including:

```text
features.parquet
metrics.csv
feature_importance.csv
best_model.joblib
synthetic_DTS_full_well.csv
geomech_fracture_gradient_holdout.csv
geomech_summary.json
run_summary.json
manifest.json
```

Diagnostic figures include:

```text
01_correlation.png
02_pred_vs_actual.png
03_depth_track.png
04_importance.png
05_mud_weight_window.png
```

These outputs provide the processed dataset, model-performance comparison, selected model, synthetic DTS predictions, feature-importance analysis, geomechanical results, and visual diagnostics.

## Project Structure

```text
.
├── research_pipeline.py
├── Research_Data.csv
├── README.md
└── results/
    ├── features.parquet
    ├── metrics.csv
    ├── feature_importance.csv
    ├── best_model.joblib
    ├── synthetic_DTS_full_well.csv
    ├── geomech_fracture_gradient_holdout.csv
    ├── geomech_summary.json
    ├── run_summary.json
    ├── manifest.json
    └── diagnostic figures
```

## Citation

If you use this repository, methodology, or implementation in academic work, please cite the repository as follows:

```bibtex
@software{dts_ml_geomechanics,
  author       = {Salman Shakib Suprova},
  title        = {Machine Learning-Based DTS Prediction and Geomechanical Analysis},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/suprovo11/dts-prediction-and-geomechanical-analysis},
  note         = {Reproducible machine-learning and geomechanics research pipeline for Well 15/9-F-1 A, Volve field}
}
```

## Acknowledgement

The computational workflow is designed as a reproducible research pipeline, with emphasis on leakage-controlled feature engineering, depth-ordered validation, comparative machine learning evaluation, and out-of-sample geomechanical analysis.

Author: Salman Shakib Suprova
Shahjalal University of Science and Technology, Sylhet



