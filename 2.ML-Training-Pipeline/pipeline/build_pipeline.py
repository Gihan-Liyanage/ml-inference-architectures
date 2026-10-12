"""Define the SageMaker training pipeline and create/update it in AWS.

    poetry run python pipeline/build_pipeline.py --stage dev --profile codewave

The bucket and execution role are read from the `ml-training-pipeline-<stage>`
CloudFormation stack (created by `sls deploy`) unless passed explicitly.
`--print-definition` prints the pipeline JSON instead of upserting it.
`--training-job` uses the separate SageMaker Training job step, overriding
TRAIN_IN_PROCESSING_JOB (needs a training-job quota).
"""
from __future__ import annotations

import argparse
import json
import logging
import os

import boto3
from botocore.exceptions import ClientError
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import FrameworkProcessor, ProcessingInput, ProcessingOutput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo, ConditionLessThanOrEqualTo
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

logger = logging.getLogger(__name__)

CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "code")

# One container version for processing, training and serving, so the model is
# always unpickled by the same scikit-learn that pickled it. 1.2-1 is the newest
# version the SDK lists for the *inference* image scope; check
# sagemaker/image_uri_config/sklearn.json before bumping it.
SKLEARN_VERSION = "1.2-1"

# TEMPORARY: train inside the Preprocess processing job (ml.t3.medium) because this
# account has no SageMaker training-job quota yet. Set to False to restore the
# separate Train step (SageMaker Training job) once the quota is approved.
TRAIN_IN_PROCESSING_JOB = True

DEFAULT_R2_THRESHOLD = 0.80
DEFAULT_RMSE_THRESHOLD = 3.5


def pipeline_name(stage: str) -> str:
    return f"mpg-training-{stage}"


def model_package_group_name(stage: str) -> str:
    return f"mpg-linear-regression-{stage}"


def build_pipeline(
    stage: str,
    bucket: str,
    role_arn: str,
    session: PipelineSession,
    train_in_processing_job: bool = TRAIN_IN_PROCESSING_JOB,
) -> Pipeline:
    name = pipeline_name(stage)

    # --- Parameters (overridable per execution; start_pipeline passes the first four)
    input_data_uri = ParameterString("InputDataUri", default_value=f"s3://{bucket}/data/train/train.csv")
    test_data_uri = ParameterString("TestDataUri", default_value=f"s3://{bucket}/data/test/mpg_test.csv")
    r2_threshold = ParameterFloat("R2Threshold", default_value=DEFAULT_R2_THRESHOLD)
    rmse_threshold = ParameterFloat("RmseThreshold", default_value=DEFAULT_RMSE_THRESHOLD)
    processing_instance_type = ParameterString("ProcessingInstanceType", default_value="ml.t3.medium")
    # ml.t3.medium isn't offered for training jobs; ml.t3.large is the smallest that is.
    # Training quotas default to 0 on new accounts - see README "Service quotas".
    training_instance_type = ParameterString("TrainingInstanceType", default_value="ml.t3.large")
    approval_status = ParameterString("ModelApprovalStatus", default_value="PendingManualApproval")

    processor = FrameworkProcessor(
        estimator_cls=SKLearn,
        framework_version=SKLEARN_VERSION,
        role=role_arn,
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{name}-processing",
        sagemaker_session=session,
    )

    if train_in_processing_job:
        # --- 1+2. Preprocess and train in one processing job (temporary, see TRAIN_IN_PROCESSING_JOB)
        step_train = ProcessingStep(
            name="PreprocessAndTrain",
            step_args=processor.run(
                code="preprocess_and_train.py",
                source_dir=CODE_DIR,
                inputs=[ProcessingInput(source=input_data_uri, destination="/opt/ml/processing/input")],
                outputs=[
                    ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
                    ProcessingOutput(output_name="model", source="/opt/ml/processing/model"),
                ],
            ),
        )
        model_data = Join(
            on="/",
            values=[step_train.properties.ProcessingOutputConfig.Outputs["model"].S3Output.S3Uri, "model.tar.gz"],
        )
        training_steps = [step_train]
    else:
        # --- 1. Preprocess: validate + clean the uploaded CSV
        step_preprocess = ProcessingStep(
            name="Preprocess",
            step_args=processor.run(
                code="preprocess.py",
                source_dir=CODE_DIR,
                inputs=[ProcessingInput(source=input_data_uri, destination="/opt/ml/processing/input")],
                outputs=[ProcessingOutput(output_name="train", source="/opt/ml/processing/train")],
            ),
        )

        # --- 2. Train
        estimator = SKLearn(
            entry_point="train.py",
            source_dir=CODE_DIR,
            framework_version=SKLEARN_VERSION,
            role=role_arn,
            instance_type=training_instance_type,
            instance_count=1,
            output_path=f"s3://{bucket}/models",
            base_job_name=f"{name}-train",
            sagemaker_session=session,
        )
        step_train = TrainingStep(
            name="Train",
            step_args=estimator.fit(
                {
                    "train": TrainingInput(
                        s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                        content_type="text/csv",
                    )
                }
            ),
        )
        model_data = step_train.properties.ModelArtifacts.S3ModelArtifacts
        training_steps = [step_preprocess, step_train]

    # --- 3. Evaluate against the fixed holdout set
    evaluation_report = PropertyFile(name="EvaluationReport", output_name="evaluation", path="evaluation.json")
    evaluation_s3_uri = Join(on="/", values=["s3:/", bucket, "evaluation", ExecutionVariables.PIPELINE_EXECUTION_ID])
    step_evaluate = ProcessingStep(
        name="Evaluate",
        step_args=processor.run(
            code="evaluate.py",
            source_dir=CODE_DIR,
            inputs=[
                ProcessingInput(source=model_data, destination="/opt/ml/processing/model"),
                ProcessingInput(source=test_data_uri, destination="/opt/ml/processing/test"),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation",
                    source="/opt/ml/processing/evaluation",
                    destination=evaluation_s3_uri,
                )
            ],
        ),
        property_files=[evaluation_report],
    )

    # --- 4a. Register (only reached when the thresholds pass)
    model = SKLearnModel(
        model_data=model_data,
        entry_point="inference.py",
        source_dir=CODE_DIR,
        framework_version=SKLEARN_VERSION,
        role=role_arn,
        sagemaker_session=session,
    )
    step_register = ModelStep(
        name="RegisterModel",
        step_args=model.register(
            content_types=["application/json"],
            response_types=["application/json"],
            inference_instances=["ml.m5.large"],
            transform_instances=["ml.m5.large"],
            model_package_group_name=model_package_group_name(stage),
            approval_status=approval_status,
            model_metrics=ModelMetrics(
                model_statistics=MetricsSource(
                    s3_uri=Join(on="/", values=[evaluation_s3_uri, "evaluation.json"]),
                    content_type="application/json",
                )
            ),
            description=Join(on=" ", values=["Trained on", input_data_uri]),
        ),
    )

    # --- 4b. Fail with a readable reason when the thresholds don't pass
    step_fail = FailStep(
        name="MetricsBelowThreshold",
        error_message=Join(
            on=" ",
            values=[
                "Model rejected: requires R2 >=", r2_threshold, "and RMSE <=", rmse_threshold,
                "- see evaluation.json for this execution",
            ],
        ),
    )

    # --- 4. Gate on the evaluation metrics
    step_condition = ConditionStep(
        name="CheckMetrics",
        conditions=[  # all conditions must hold
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(step_name=step_evaluate.name, property_file=evaluation_report,
                             json_path="regression_metrics.r2.value"),
                right=r2_threshold,
            ),
            ConditionLessThanOrEqualTo(
                left=JsonGet(step_name=step_evaluate.name, property_file=evaluation_report,
                             json_path="regression_metrics.rmse.value"),
                right=rmse_threshold,
            ),
        ],
        if_steps=[step_register],
        else_steps=[step_fail],
    )

    return Pipeline(
        name=name,
        parameters=[
            input_data_uri, test_data_uri, r2_threshold, rmse_threshold,
            processing_instance_type, training_instance_type, approval_status,
        ],
        steps=[*training_steps, step_evaluate, step_condition],
        sagemaker_session=session,
    )


def ensure_model_package_group(sagemaker_client, group_name: str) -> None:
    try:
        sagemaker_client.describe_model_package_group(ModelPackageGroupName=group_name)
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ValidationException":
            raise
        logger.info("Creating model package group %s", group_name)
        sagemaker_client.create_model_package_group(
            ModelPackageGroupName=group_name,
            ModelPackageGroupDescription="Auto MPG multiple linear regression",
        )


def stack_outputs(boto_session: boto3.Session, stage: str) -> dict:
    cfn = boto_session.client("cloudformation")
    stack = cfn.describe_stacks(StackName=f"ml-training-pipeline-{stage}")["Stacks"][0]
    return {o["OutputKey"]: o["OutputValue"] for o in stack.get("Outputs", [])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", default="dev")
    parser.add_argument("--region", default="eu-west-1")
    parser.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    parser.add_argument("--bucket", help="Training bucket (default: from stack outputs)")
    parser.add_argument("--role-arn", help="SageMaker execution role (default: from stack outputs)")
    parser.add_argument("--print-definition", action="store_true", help="Print the JSON definition and exit")
    parser.add_argument(
        "--training-job",
        action="store_true",
        help="Train in a SageMaker Training job instead of inside the processing job",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    boto_session = boto3.Session(profile_name=args.profile, region_name=args.region)

    if not (args.bucket and args.role_arn):
        outputs = stack_outputs(boto_session, args.stage)
        args.bucket = args.bucket or outputs["TrainingBucketName"]
        args.role_arn = args.role_arn or outputs["SageMakerExecutionRoleArn"]

    session = PipelineSession(
        boto_session=boto_session,
        default_bucket=args.bucket,
        default_bucket_prefix="sagemaker",
    )
    pipeline = build_pipeline(
        args.stage, args.bucket, args.role_arn, session,
        train_in_processing_job=TRAIN_IN_PROCESSING_JOB and not args.training_job,
    )

    if args.print_definition:
        print(json.dumps(json.loads(pipeline.definition()), indent=2))
        return

    ensure_model_package_group(session.sagemaker_client, model_package_group_name(args.stage))
    response = pipeline.upsert(
        role_arn=args.role_arn,
        description="Auto MPG: preprocess -> train -> evaluate -> threshold gate -> register",
    )
    logger.info("Upserted pipeline %s", response["PipelineArn"])


if __name__ == "__main__":
    main()
