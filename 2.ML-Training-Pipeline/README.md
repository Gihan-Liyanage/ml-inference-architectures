# ML Training Pipeline: SageMaker Pipelines

Automated retraining for the Auto MPG multiple linear regression model from [`1.ML-Serverless-Inference`](../1.ML-Serverless-Inference). Uploading a training CSV to S3 starts the pipeline:

1. Train the model.
2. Evaluate it on a fixed holdout set.
3. Gate it on R² / RMSE thresholds.
4. Register it in the SageMaker Model Registry, where it waits for **manual approval**.

Once you approve the model, it is deployed to a **SageMaker Serverless Inference endpoint**.

## Architecture

```
aws s3 cp train.csv s3://codewave-<stage>-ml-training/data/train/<name>.csv
   │  (bucket sends events to EventBridge)
   ▼
EventBridge rule: Object Created, key "data/train/*.csv"     ── failures ──▶ SQS failed-events queue
   ▼
Lambda startPipeline ─▶ StartPipelineExecution(InputDataUri, R2Threshold, RmseThreshold)
   ▼
SageMaker Pipeline "mpg-training-<stage>"
   Preprocess ─▶ Train ─▶ Evaluate ─▶ CheckMetrics (R² ≥ min AND RMSE ≤ max)
                                         ├─ pass ─▶ RegisterModel (PendingManualApproval)
                                         └─ fail ─▶ MetricsBelowThreshold (Fail step)
   ▼
SageMaker Model Registry: group "mpg-linear-regression-<stage>"
   ├─ PendingManualApproval ─▶ SNS email "waiting for approval"
   └─ Approved (by you) ─────▶ Lambda deployModel ─▶ Serverless endpoint "mpg-<stage>-serverless"

Any failed pipeline execution ─▶ SNS email
```

The pipeline steps share one code bundle in [`pipeline/code/`](pipeline/code):

| Step | Script | Input | Output |
|---|---|---|---|
| Preprocess | `preprocess.py` | uploaded CSV(s) | cleaned `train.csv` (fails on missing columns or < 50 usable rows) |
| Train | `train.py` | cleaned CSV | `model.tar.gz` containing `model.joblib` |
| Evaluate | `evaluate.py` | model + `data/test/mpg_test.csv` | `evaluation.json` (`regression_metrics.r2/rmse/mae`) |
| Serve | `inference.py` | JSON request | `{"predicted_mpg": [...]}` |

`common.py` holds the column lists, cleaning and the model definition taken from the notebook. Every step imports it, so training and serving cannot drift.

## Architecture Decision Records (ADR)

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | **SageMaker Pipelines** orchestrates train → evaluate → gate → register | Purpose-built for ML workflows. Per-run lineage, the Model Registry (versions, metrics, approval status) and a visual run graph in Studio come for free. Step Functions would be cheaper and faster for a dataset this small, but would need a hand-built registry. |
| 2 | **Serverless Framework owns the infrastructure; `pipeline/build_pipeline.py` owns the pipeline definition** | The pipeline is defined with the SageMaker Python SDK, which uploads the step code and renders the pipeline JSON. That doesn't fit in CloudFormation. Bucket, roles, Lambdas, rules, SNS and SQS stay in `serverless.yml` like project 1. |
| 3 | **One prebuilt SKLearn container version (`1.2-1`) for processing, training and serving** | The model is always unpickled by the same scikit-learn that pickled it. `1.2-1` is the newest version the SDK lists for the *inference* scope. Project 1 pins `scikit-learn==1.9.0`, so this pipeline's models are **not** interchangeable with project 1's joblib files. That's fine because the deploy targets are separate. |
| 4 | **SageMaker Python SDK v2 (`>=2.257,<3`)** | v3 restructured the training and pipeline APIs. v2 is the well-documented `sagemaker.workflow` API. It prints a deprecation notice; set `SAGEMAKER_SUPPRESS_V2_WARNING=1` to hide it. |
| 5 | **Fixed holdout test set** at `data/test/mpg_test.csv`, made once by `scripts/split_holdout.py` | Every run is scored on the same rows, so metrics are comparable across runs and the threshold means the same thing each time. The trigger only watches `data/train/`, so the test file never starts a run. **Don't put holdout rows into later training files**, or scores will be inflated. |
| 6 | **Metric gate: R² ≥ 0.80 and RMSE ≤ 3.5 mpg** (both must hold) | "Accuracy" doesn't apply to regression. The notebook's model scores R² 0.827 / RMSE 2.95 on the same split. The values live in `serverless.yml` (`custom.r2Threshold`, `custom.rmseThreshold`) and are passed to every execution, so changing them only needs `sls deploy`. |
| 7 | **Manual approval through the Model Registry, not inside the pipeline** | The pipeline finishes once it registers a `PendingManualApproval` version. Approval is a status change on that version, made by anyone with `sagemaker:UpdateModelPackage`. It emits an EventBridge event that triggers the deployment. No pipeline sits waiting for days. |
| 8 | **EventBridge (not SQS) as the event backbone; SQS only as the failed-events queue** | SageMaker publishes registry and pipeline status changes only to EventBridge, so the S3 trigger uses it too for consistency. EventBridge also supports key wildcard filtering. One upload means one run, so there's nothing to buffer. |
| 9 | **`startPipeline` is fire-and-forget and idempotent** | It returns as soon as the execution is queued. `ClientRequestToken` = hash(bucket, key, S3 version-id): retries of the same event can't start duplicate runs, while re-uploading the same file *does* retrain. EventBridge retries for up to 1 h; events that still fail land in the SQS queue. |
| 10 | **Deploy via `Containers=[{ModelPackageName}]` with fresh model/config names and a fixed endpoint name** | The endpoint runs exactly the approved package (image + artifact + `inference.py`). Models and endpoint configs are immutable, so each deploy creates new ones and calls `UpdateEndpoint`, a blue/green swap with no downtime. |
| 11 | **Pipeline execution role uses `AmazonSageMakerFullAccess` + an inline policy for the training bucket** | Keeps the role simple for a solo project; the Lambda roles are least-privilege. Narrow the execution role before using this in production. |
| 12 | **`mpg.csv` is vendored in `seed-data/`** | Project 1's `scripts/` folder is gitignored, so the dataset is copied here to make the tests and the split script work from a clean clone. |
| 13 | **CI/CD with GitHub Actions + OIDC; one workflow runs `sls deploy` then `build_pipeline.py`** | The repo already lives on GitHub, so there's no second CI system to run. OIDC gives short-lived AWS credentials, with no stored keys. The order matters: the pipeline upsert reads the bucket and role from the stack that `sls deploy` just created or updated. The upsert is idempotent, so running it on every deploy is safe. |

## Service quotas

Each run needs one `ml.t3.medium` **processing** instance and one `ml.t3.large` **training** instance. New AWS accounts start with a training quota of **0** for every instance type, so request 1 before the first run. Quotas are per region.

```bash
aws service-quotas request-service-quota-increase --service-code sagemaker \
  --quota-code L-B95AAE44 --desired-value 1 --region eu-west-1    # ml.t3.large for training job usage
aws service-quotas get-service-quota --service-code sagemaker \
  --quota-code L-B95AAE44 --region eu-west-1 --query "Quota.Value" # 1.0 once approved
```

Without it, the Train step fails with `ResourceLimitExceeded`. To use a different training instance type, change the `TrainingInstanceType` default in `pipeline/build_pipeline.py` and re-run the upsert. Pick a type whose training quota you've been granted.

## Deployment

> **Requirements:** Node (see `.nvmrc`), Poetry, AWS CLI with a profile for the target account.

```bash
npm install
poetry install

# 1. Infrastructure (bucket, roles, Lambdas, EventBridge rules, SNS, SQS)
npx sls deploy --stage dev --aws-profile codewave --param="alertEmail=you@example.com"
#    → confirm the SNS subscription email AWS sends you

# 2. Pipeline definition (reads the bucket + role from the stack outputs)
poetry run python pipeline/build_pipeline.py --stage dev --profile codewave
#    Re-run this whenever pipeline/ changes. --print-definition shows the JSON instead.

# 3. Fixed holdout set (once)
poetry run python scripts/split_holdout.py        # → data/train.csv, data/mpg_test.csv
aws s3 cp data/mpg_test.csv s3://codewave-dev-ml-training/data/test/mpg_test.csv --profile codewave
```

`sls package` warns that `python3.12` isn't in Serverless v3's runtime list. The warning is harmless: CloudFormation and Lambda accept the runtime.

## CI/CD (GitHub Actions)

[`.github/workflows/ml-training-pipeline.yml`](../.github/workflows/ml-training-pipeline.yml) automates steps 1–2 above. It only runs when `2.ML-Training-Pipeline/**` or the workflow itself changes.

| Trigger | What runs |
|---|---|
| Pull request | `pytest` |
| Push to `development` | `pytest` → `sls deploy --stage dev` → `build_pipeline.py --stage dev` |
| Manual (*Actions → ml-training-pipeline → Run workflow*) | Same as the push row, for the chosen stage (`dev` or `prod`) |

Deploys to the same stage are queued, never run in parallel. The workflow authenticates to AWS with **GitHub OIDC**, so no AWS keys are stored in GitHub. Step 3, uploading the holdout test set, is still a one-off manual step per stage.

### One-time setup

1. **Create the deploy role** that GitHub assumes. Do this once per AWS account.
   ```bash
   aws cloudformation deploy --profile codewave --region eu-west-1 \
     --template-file ci/github-oidc-role.yml \
     --stack-name github-actions-ml-training-pipeline \
     --capabilities CAPABILITY_NAMED_IAM
   # Add --parameter-overrides CreateOidcProvider=false if the account already has
   # the token.actions.githubusercontent.com OIDC provider (only one is allowed).
   aws cloudformation describe-stacks --stack-name github-actions-ml-training-pipeline \
     --query "Stacks[0].Outputs" --profile codewave --region eu-west-1
   ```
2. **Create the GitHub Environments.** In GitHub, go to *Settings → Environments* and create `dev` and `prod`. The role only trusts jobs running in these two environments.
   - Set these variables on each environment:
     - `AWS_DEPLOY_ROLE_ARN`: the `DeployRoleArn` output from step 1.
     - `ALERT_EMAIL`: where the approval and failure emails go.
   - Recommended protection: limit both environments to the `development` branch, and add **required reviewers** on `prod` so a prod deploy waits for approval.

The role's permissions are scoped to this stack's resource names (`ml-training-pipeline-*`, `codewave-*-ml-training`, `mpg-training-*`, `mpg-linear-regression-*`). It can still create IAM roles with that prefix, which `sls deploy` needs, so treat the `dev`/`prod` environment protection as part of the security boundary. If a deploy fails with `AccessDenied`, add the action named in the error to `ci/github-oidc-role.yml` and redeploy that stack.

## Run it

```bash
aws s3 cp data/train.csv s3://codewave-dev-ml-training/data/train/train-$(date +%s).csv --profile codewave
```

Follow the run in SageMaker Studio (Pipelines → `mpg-training-dev`), or with the CLI:

```bash
aws sagemaker list-pipeline-executions --pipeline-name mpg-training-dev --profile codewave
aws sagemaker list-pipeline-execution-steps --pipeline-execution-arn <arn> --profile codewave
```

A run takes roughly 10–15 minutes, mostly spent provisioning the jobs. If the metrics miss the gate, the run fails at `MetricsBelowThreshold` and you get an email. To test the gate on purpose, start a run with a stricter threshold:

```bash
aws sagemaker start-pipeline-execution --pipeline-name mpg-training-dev \
  --pipeline-parameters Name=R2Threshold,Value=0.99 --profile codewave
```

## Approve and deploy

When a version passes the gate, you get an email with its ARN. Review its metrics in Studio: Models → Model registry → `mpg-linear-regression-dev` → the version → Model quality. Then approve it there with **Update status → Approved**, or from the CLI:

```bash
aws sagemaker list-model-packages --model-package-group-name mpg-linear-regression-dev --profile codewave
aws sagemaker update-model-package --model-package-arn <arn> \
  --model-approval-status Approved --profile codewave
```

Approval triggers `deployModel`. The first approval creates the endpoint and later ones update it in place. Status `Rejected` deploys nothing. Wait for `InService`:

```bash
aws sagemaker describe-endpoint --endpoint-name mpg-dev-serverless \
  --query EndpointStatus --profile codewave
```

## Invoke the endpoint

```bash
aws sagemaker-runtime invoke-endpoint --endpoint-name mpg-dev-serverless \
  --content-type application/json --cli-binary-format raw-in-base64-out \
  --body '{"displacement":140,"horsepower":90,"weight":2264,"acceleration":15.5,"model_year":76,"origin":"usa"}' \
  --profile codewave out.json && cat out.json
# {"predicted_mpg": [25.3...]}
```

The body can also be a list of such objects, which returns one prediction each. The endpoint is reachable with IAM (SigV4) credentials only. A public HTTPS API in front of it (API Gateway → `InvokeEndpoint`) isn't part of this project yet. Serverless endpoints scale to zero, so the first request after idle has a cold start of a few seconds.

## Troubleshooting

- **Upload didn't start a run:** check that the key matches `data/train/*.csv`, check the `startPipeline` logs, and look in the `ml-training-pipeline-<stage>-failed-events` SQS queue.
- **Approved but not deployed:** check the `deployModel` logs. `UpdateEndpoint` fails while the endpoint is still `Creating`/`Updating`; EventBridge retries it for up to 1 h, after which the event lands in the failed-events queue. To replay it, save the message body as `event.json` and run `npx sls invoke -f deployModel --path event.json --stage dev`.
- **Orphaned models/endpoint configs** accumulate across deploys; there's no automatic cleanup yet.

## Tests

```bash
poetry run pytest
```

The suite covers:
- The scripts end to end on the notebook's split, which should reproduce R² ≈ 0.827 and RMSE ≈ 2.95.
- The inference handlers.
- Both Lambdas, with stubbed SageMaker calls.
- The compiled pipeline definition, generated offline with `moto`. No AWS calls are made.
