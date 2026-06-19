# Cambridge Energy Rating Predictor — Current Notebook Results

This package keeps the original Streamlit visual design while synchronising the website with the uploaded Colab notebook `Final_notebook_by_pinky.ipynb`. Its 28 code cells and all saved code-cell outputs are identical to the bundled source notebook `FINAL_READY_FILE_NO_MISSING_CLEANING(6).ipynb`.

## Run locally

From inside the `APP` folder:

```bash
streamlit run app.py
```

For Streamlit Community Cloud, use the main file path:

```text
Cambridge_EPC_Final_Code/APP/app.py
```

## Current notebook results

- Records: **5,000**
- Original variables: **93**
- Approved model predictors: **38**
- Model-ready CSV: **5,000 rows × 43 columns**
- Best classifier: **Random Forest Classifier**
- Held-out accuracy: **0.7580**
- Weighted F1: **0.7576**
- Macro F1: **0.5235**
- Balanced accuracy: **0.5305**
- Best regressor: **Linear Regression**
- Held-out R²: **0.7687**
- MAE: **3.3244**
- RMSE: **4.9398**

## Colab alignment correction

- CSV loading preserves the legitimate engineered category **`GLAZING_GROUP = "None"`** for 122 properties. Pandas previously interpreted that word as a missing value, which changed live model inputs.
- Headline model metrics are loaded from `CURRENT_NOTEBOOK_RESULTS.json`, the executed notebook source of truth.
- Month/year evaluation uses only the exact 1,000 held-out classification predictions exported from Colab; training records are not mixed into evaluation.
- The included `validate_colab_streamlit_alignment.py` script checks notebook code/outputs, result tables, diagnostics, feature importance, model inputs, predictions and confusion matrix.

The latest comparison report records **38 passed checks and 0 failures**.

## Leakage control

The prediction models use only independent physical and structural property variables. Current/potential EPC outputs, energy consumption, CO₂, costs, environmental-impact outputs, assessment labels and target-encoded variables are excluded from model inputs.

## Main website updates

- The Welcome page now shows **38 features** and **75.8% accuracy**.
- The model-performance page uses the current cross-validation and held-out metrics.
- The selected regressor is **Linear Regression**, not Random Forest Regressor.
- The feature-importance page uses notebook permutation importance and the correct leading variables.
- The full original three-stage Prediction dashboard is restored with current-period graphs, all-property comparison cards, planning-scenario graphs and summary tables.
- All four property types are selected by default on Prediction and Feature Explorer, and the Interactive Dashboard opens with **All property types** selected.
- The prediction inputs build the same 38 leakage-free features as the notebook.
- Energy, CO₂ and cost charts in the planning section are clearly labelled as historical reference medians, not model inputs or future forecasts.
- Recommendations and explanatory text no longer describe leakage variables as model drivers.
- The original website layout, styling and navigation are retained.

## Included files

```text
APP/
├── app.py
├── requirements.txt
├── .streamlit/config.toml
├── data/
├── models/
└── source_files/
    ├── FINAL_READY_FILE_NO_MISSING_CLEANING(6).ipynb
    └── COM6003_Cambridge_Energy_Rating_Predictor_FINAL_FORMATTED_EXCEL.ipynb
```

## Deployment note

If an existing Streamlit deployment still shows **47 features** or **94.7% accuracy**, it is serving the previous repository version. Replace the repository files with this package, commit/push the changes, then reboot or redeploy the Streamlit app.
