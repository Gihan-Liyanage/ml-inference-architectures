# ml-inference-architectures
This repository contains examples of difference inference architectures for machine learning models.

Both projects serve the same model: a scikit-learn multiple linear regression that predicts a car's fuel efficiency (`mpg`) from the Auto MPG dataset. Each project's README has its architecture diagram, design decisions and deployment steps.

| | [1. ML Serverless Inference](1.ML-Serverless-Inference) | [2. ML Training Pipeline](2.ML-Training-Pipeline) |
|---|---|---|
| **Purpose** | Serve a trained model behind a public HTTP API | Retrain, evaluate, approve and deploy the model automatically |
| **Trigger** | HTTPS `POST /predict` | CSV upload to `s3://codewave-<stage>-ml-training/data/train/` |
| **Training** | Manual, in the notebook | SageMaker Pipeline: preprocess + train → evaluate on a fixed holdout set |
| **Quality gate** | None | R² ≥ 0.80 and RMSE ≤ 3.5, then manual approval in the Model Registry |
| **Model storage** | `joblib` file uploaded by hand to `s3://codewave-<stage>-ml-models` | Versioned in the SageMaker Model Registry (`mpg-linear-regression-<stage>`) |
| **Serving** | Lambda container image (arm64) loading the model from S3 | SageMaker Serverless Inference endpoint (`mpg-<stage>-serverless`) |
| **API access** | Public HTTP API (API Gateway) | IAM-authenticated `InvokeEndpoint` (SigV4) |
| **Key AWS services** | API Gateway, Lambda, ECR, S3, CloudWatch | S3, EventBridge, Lambda, SageMaker Pipelines + Model Registry + Serverless Inference, SNS, SQS, CloudWatch |
| **Notifications** | — | Email (SNS) when a model awaits approval or a run fails |
| **Infrastructure as code** | Serverless Framework (`serverless.yml`) | Serverless Framework + `build_pipeline.py` (SageMaker SDK) |
| **CI/CD** | — | GitHub Actions with OIDC: tests on PRs, deploy on push to `development` |
| **Idle cost** | ~$0 (pay per request) | ~$0 (pay per run and per request) |
