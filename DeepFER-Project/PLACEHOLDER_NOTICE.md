# ⚠️ Placeholder Model Artifacts

The `best_model.keras`, `class_names.pkl`, and `model_config.json` in this folder are from a
**2-epoch smoke test on a small subset** — used only to validate that the full pipeline (training,
saving, and app inference) works end-to-end. They are NOT a meaningfully trained model.

Run `notebooks/DeepFER_Stage2_Training.ipynb` on Kaggle with `FAST_DEV_RUN = False` to train for
real, then replace these files with the resulting `best_model.keras`, `class_names.pkl`,
`label_encoder.pkl`, `training_history.pkl`, and `model_config.json` from your Kaggle output.
