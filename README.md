# Bank Churn Prediction on Microsoft Fabric

This project is a small end-to-end machine learning workflow for predicting bank customer churn in Microsoft Fabric. It starts with raw customer data, cleans and explores it in a Lakehouse, trains a few classification models, registers the best model with MLflow, and then uses that registered model for batch scoring.

The project is written as a notebook-first Fabric workflow, so the main story lives in the notebooks rather than in standalone Python modules.

## What This Project Does

The workflow uses a bank churn dataset with customer attributes such as credit score, geography, gender, age, tenure, balance, number of products, credit card ownership, active membership status, and estimated salary.

The goal is to predict whether a customer is likely to leave the bank.

The current modeling flow trains and compares:

- Random Forest models
- A LightGBM classifier

The exported model included in this repository is `lgbm_sm`, saved in MLflow format.

## Repository Structure

```text
.
+-- notebooks/
|   +-- 1-ingest_data.ipynb
|   +-- 2-explore-cleanse-data.ipynb
|   +-- 3-Train-register-MLmodel.ipynb
|   +-- 4-Predict.ipynb
+-- models/
|   +-- lgbm_sm/
|       +-- MLmodel
|       +-- model.pkl
|       +-- conda.yaml
|       +-- python_env.yaml
|       +-- requirements.txt
+-- reports/
```

## Notebook Flow

Run the notebooks in order:

1. **Ingest data**

   Downloads the demo churn dataset into the Fabric Lakehouse under `Files/churn/raw`.

2. **Explore and cleanse data**

   Loads the raw CSV, explores the data, applies cleaning and feature engineering, and writes the cleaned dataset to a Delta table named `df_clean`.

3. **Train and register ML models**

   Loads `df_clean`, creates train/validation/test splits, handles class imbalance with SMOTE, trains Random Forest and LightGBM models, logs metrics with MLflow, and registers the trained models.

4. **Predict**

   Loads the test data from the Lakehouse, uses the registered `lgbm_sm` model for batch scoring, and writes predictions back to a Delta table.

## Model

The included model artifact is:

```text
models/lgbm_sm
```

It is an MLflow LightGBM model with the following key runtime dependencies:

- Python 3.10.12
- MLflow 2.12.2
- LightGBM 4.0.0
- scikit-learn 1.6.1
- pandas 2.0.3
- numpy 1.26.4
- scipy 1.15.3

The model signature expects engineered and one-hot encoded columns such as `Geography_France`, `Geography_Germany`, `Gender_Female`, `NewTenure`, `NewCreditsScore`, and similar features created during the cleaning step.

## Running in Microsoft Fabric

This project assumes a Fabric workspace with a Lakehouse attached to the notebooks.

Before running the full flow, make sure:

- The notebooks are attached to a Fabric Lakehouse.
- The default Lakehouse path `/lakehouse/default` is available.
- The Spark kernel is `synapse_pyspark`.
- The MLflow experiment and registered model names can be created in your workspace.

The prediction notebook expects a registered model named:

```text
lgbm_sm
```

and currently uses version:

```text
1
```

If you retrain and register a new version, update the model version in the prediction notebook before scoring.

## Notes

This is a practical learning project rather than a production service. The notebooks intentionally show the full process step by step: data loading, exploration, feature engineering, model training, registration, and scoring.

For a more production-ready version, the next useful improvements would be:

- Move shared preprocessing logic into reusable functions.
- Keep dependency versions in one place.
- Clear notebook outputs before committing.
- Add basic validation checks for the expected input schema.
- Package the feature engineering and model inference flow together so training and scoring always use the same transformations.

## Dataset

The demo dataset is downloaded from a public Microsoft-hosted sample URL inside the ingestion notebook. If you want to use your own data, set `IS_CUSTOM_DATA = True` in the ingestion step and place your CSV in the expected Lakehouse location.
