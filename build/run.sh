pip install --upgrade --force-reinstall . "awscli>1.20.30"
export ARTIFACT_BUCKET=mnist-mlops-v1
export AWS_REGION=eu-west-1
export SAGEMAKER_PIPELINE_ROLE_ARN=arn:aws:iam::367216843975:role/a204384-sagemaker-tpa
export SAGEMAKER_PROJECT_ID=v1
export SAGEMAKER_PROJECT_NAME=mnist-mlops
export SAGEMAKER_PROJECT_NAME_ID="${SAGEMAKER_PROJECT_NAME}-${SAGEMAKER_PROJECT_ID}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
run-pipeline --module-name pipelines.mnist.pipeline --role-arn $SAGEMAKER_PIPELINE_ROLE_ARN --tags "[{\"Key\":\"sagemaker:project-name\", \"Value\":\"${SAGEMAKER_PROJECT_NAME}\"}, {\"Key\":\"sagemaker:project-id\", \"Value\":\"${SAGEMAKER_PROJECT_ID}\"}]" --kwargs "{\"region\":\"${AWS_REGION}\",\"role\":\"${SAGEMAKER_PIPELINE_ROLE_ARN}\",\"default_bucket\":\"${ARTIFACT_BUCKET}\",\"pipeline_name\":\"${SAGEMAKER_PROJECT_NAME_ID}\",\"model_package_group_name\":\"${SAGEMAKER_PROJECT_NAME_ID}\",\"base_job_prefix\":\"${SAGEMAKER_PROJECT_NAME_ID}\"}"