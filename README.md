# Cambridge Energy Rating Predictor — Current Notebook Results

This package keeps the original Streamlit visual design while synchronising the entire website with `FINAL_READY_FILE_NO_MISSING_CLEANING(6).ipynb`.

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

## Leakage control

The prediction models use only independent physical and structural property variables. Current/potential EPC outputs, energy consumption, CO₂, costs, environmental-impact outputs, assessment labels and target-encoded variables are excluded from model inputs.

## Main website updates

- The Welcome page now shows **38 features** and **75.8% accuracy**.
- The model-performance page uses the current cross-validation and held-out metrics.
- The selected regressor is **Linear Regression**, not Random Forest Regressor.
- The feature-importance page uses notebook permutation importance and the correct leading variables.
- The prediction form now builds the same 38 leakage-free features as the notebook.
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
