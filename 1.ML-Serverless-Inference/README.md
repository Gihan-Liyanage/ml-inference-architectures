# Serverless - AWS Python Docker

This project has been generated using the `aws-nodejs-docker` template from the [Serverless framework](https://www.serverless.com/).

For detailed instructions, please refer to the [documentation](https://www.serverless.com/framework/docs/providers/aws/).

## Deployment instructions

> **Requirements**: Docker. In order to build images locally and push them to ECR, you need to have Docker installed on your local machine. Please refer to [official documentation](https://docs.docker.com/get-docker/).

In order to deploy your service, run the following command

```
sls deploy
```

This also creates the `codewave-<stage>-ml-models` S3 bucket via CloudFormation, but does **not** upload the trained model into it — that's a separate, manual step (see below).

## Upload the model to S3

After the first deploy (or whenever the model is retrained/re-exported from the notebook), upload the `.joblib` artifact to the bucket the stack created:

```
aws s3 cp scripts/mlr_mpg_v1.joblib \
  s3://codewave-dev-ml-models/models/mlr_mpg_v1.joblib \
  --profile codewave
```

Replace `dev` with the target stage if deploying elsewhere. The Lambda loads this object lazily on first invocation and caches it in memory for the life of the execution environment, so a redeploy (or waiting for the container to recycle) is needed to pick up a newly uploaded model.

## Test your service

After successful deployment and uploading the model, call the `/predict` endpoint:

```
curl -X POST https://<api-id>.execute-api.eu-west-1.amazonaws.com/predict \
  -H 'Content-Type: application/json' \
  -d '{"displacement":140,"horsepower":90,"weight":2264,"acceleration":15.5,"model_year":76,"origin":"usa"}'
```

or invoke it directly:

```
sls invoke --function predict --data '{"body": "{\"displacement\":140,\"horsepower\":90,\"weight\":2264,\"acceleration\":15.5,\"model_year\":76,\"origin\":\"usa\"}"}'
```
