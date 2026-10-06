# Fake Instagram Profile Detector — Verified Streamlit

This Streamlit prototype uses the bundled XGBoost model retrained with 500 trees.

## Training run

- Original rows: 5,000
- Duplicate rows removed: 66
- Clean rows: 4,934
- Training rows: 3,947
- Test rows: 987
- Split: stratified 80/20 train/test, `random_state=42`
- Training: one fixed configuration; no grid search or cross-validation

### Model configuration

- `n_estimators=500`
- `max_depth=5`
- `learning_rate=0.1`
- `subsample=1.0`
- `colsample_bytree=0.8`
- `objective=binary:logistic`
- `random_state=42`

### Held-out test result from the supplied retraining run

- Accuracy: 99.39%
- Precision: 99.59%
- Recall: 99.19%
- F1: 99.39%
- ROC-AUC: 0.9995

The application loads the fitted `research_model.joblib` artifact. It does **not** retrain the model when a user opens the app. The displayed metrics and model parameters come from the supplied retraining bundle.

### Previously reported tuned-model confusion matrix

The research panel also shows the earlier tuned XGBoost model's confusion matrix: `[[493, 2], [8, 484]]` (2 genuine profiles flagged as fake; 8 fake profiles missed). This is a separate prior result and does not describe the currently bundled 500-tree model.

## Features

- Manual single-profile screening
- Batch CSV screening
- Adjustable decision threshold
- Model-level XGBoost contribution display
- Research verification panel
- Downloadable batch results
- Clear distinction between a model score and a calibrated real-world probability

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app does not require the research CSV to make predictions because the fitted model artifact is bundled. For batch evaluation, include the optional `fake` column.
