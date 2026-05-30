# Cambridge Energy Rating Predictor - Streamlit App

This is an advanced dark-theme Streamlit dashboard scaffold for the COM6003 Cambridge Energy Rating Predictor project.

## Files
- `app.py` — main Streamlit application
- `.streamlit/config.toml` — dark theme settings
- `requirements.txt` — required packages
- `data/cleaned_energy_data.csv` — place your cleaned dataset here
- `models/best_model.pkl` — place your trained model here later

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Important
If `data/cleaned_energy_data.csv` is not found, the app uses demo data so that the dashboard still runs.

The prediction page currently has a fallback scoring logic. After your final model file is available, we can connect the real trained model.
