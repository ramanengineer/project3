# NZ Road Crash Severity Predictor
## 297.201 Project 3 — Waka Kotahi CAS Data

### Setup

```bash
pip install -r requirements.txt
```

### How to run

**Step 1 — Run the notebook**
```
jupyter notebook project3_notebook.ipynb
```
Run all cells top to bottom. This will:
- Clean and process the CAS data
- Train 3 ML models
- Save `best_model.pkl`, `label_encoder.pkl`, `rf_model.pkl`, `feature_meta.json`, `feature_columns.json`

**Step 2 — Launch the web app**
```
streamlit run app.py
```
Open the URL shown (usually http://localhost:8501)

### Files
- `project3_notebook.ipynb` — Main analysis notebook
- `app.py` — Streamlit web application
- `cas_data.csv` — Raw data (from Waka Kotahi open data)
- `contribution_statement.md` — Team contribution document
- `requirements.txt` — Python dependencies
