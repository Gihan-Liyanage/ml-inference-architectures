import os

# src.config reads these eagerly at import time; set test defaults before any
# src.* module is imported by the test collection process.
os.environ.setdefault("MODEL_BUCKET_NAME", "test-bucket")
os.environ.setdefault("MODEL_OBJECT_KEY", "models/mlr_mpg_v1.joblib")
