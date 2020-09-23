# Documentation

| Module | Function | Description | Parameters | Yields | Returns |
| :--- | :--- | :--- | :--- | :--- | :--- |
| preprocess | make_dataset() | Performs pre-processing | raw_file_name | train_set, train_clean, test_set, test_clean & ord_encoder.pkl | train_clean, test_clean
| train | train_model() | Trains LightBGM Model on train_clean | -- | lead_scoring_model.pkl | training cross_validation results.
| predict | test_model() | Predicts on the test_clean & Segments Leads into High, Medium, Low Probability Categories based on model prediction  | thershold=0.35 | lead_scoring.csv | classification_report, confusion_matrix(normalize="true")
