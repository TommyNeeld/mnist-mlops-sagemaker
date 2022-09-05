import argparse
import logging
import os

import boto3
import sagemaker
from sagemaker.model import ModelPackage
import sagemaker.session


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def _get_model_config(boto3_client_sm, model_package_name):
    models = boto3_client_sm.list_model_packages(
        ModelPackageType="Versioned", ModelPackageGroupName=model_package_name
    )["ModelPackageSummaryList"]

    if len(models) == 0:
        return {"statusCode": 400, "body": f"No models found for model package: {model_package_name}"}

    package_version = 0
    model_arn = None
    for model in models:
        if (model["ModelPackageVersion"] > package_version) and (model["ModelApprovalStatus"] == "Approved"):
            package_version = model["ModelPackageVersion"]
            model_arn = model["ModelPackageArn"]

    if not model_arn:
        return {"statusCode": 400, "body": f"Latest Approved model could not be identified for group: {models}"}

    # get instance type
    describe_model = boto3_client_sm.describe_model_package(ModelPackageName=model_arn)
    instance_type = describe_model["InferenceSpecification"]["SupportedTransformInstanceTypes"][0]
    model_name = f'{describe_model["ModelPackageGroupName"]}-{describe_model["ModelPackageVersion"]}'

    return instance_type, model_name, model_arn


def main():
    """Create and run the Batch Transform Job"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="eu-west-1")
    parser.add_argument("--s3-bucket", default=None)
    parser.add_argument("--s3-data-path", default=None)
    parser.add_argument("--model-package-name", default=None)
    parser.add_argument("--model-execution-role-arn", default=None)
    args, _ = parser.parse_known_args()

    if args.s3_data_path is None:
        raise ValueError("The 's3-data-path' needs to be provided!")

    # get model config
    boto3_client_sm = boto3.client("sagemaker")
    instance_type, model_name, model_arn = _get_model_config(boto3_client_sm, args.model_package_name)
    logger.info(f"model_name: {model_name}, model_arn: {model_arn}, instance_type: {instance_type}")

    # get model object
    boto_session = boto3.Session(region_name=args.region)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    model = ModelPackage(
        model_package_arn=model_arn,
        role=args.model_execution_role_arn,
        name=model_name,
        sagemaker_session=sagemaker_session,
    )

    input_path = "s3://{}/{}/".format(args.s3_bucket, args.s3_data_path)
    output_path = "s3://{}/output-data/".format(args.s3_bucket)

    # Run the SageMaker Batch Transform Job on this model and wait for it to complete.
    # This will also create a Model object in AWS Console / SageMaker / Inference / Models
    # see https://sagemaker.readthedocs.io/en/stable/api/inference/model.html#sagemaker.model.Model.transformer
    logger.info(
        f"Starting the Batch Transform job for model '{model.name}' with input path '{input_path}' "
        f"and output path '{output_path}'"
    )
    transformer = model.transformer(
        instance_count=1,
        instance_type=instance_type,
        output_path=output_path,
        strategy="MultiRecord",
        assemble_with="Line",
    )
    # see model handler docs here...
    # https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/using_tf.html#how-to-implement-the-pre-and-or-post-processing-handler-s
    transformer.transform(input_path, content_type="application/jsonlines", split_type="Line")
    transformer.wait()

    logger.info("The Batch Transform job has completed.")

    # optional - deleting the model (the ModelPackage stays, only the model is deleted)
    logger.info(f"Deleting the model '{model.name}'")
    model.delete_model()


if __name__ == "__main__":
    main()
