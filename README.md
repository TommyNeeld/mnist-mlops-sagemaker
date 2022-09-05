# mnist-mlops
Training and batch inference code for MNIST model

## Problem

Code to run model training and inference for the MNIST model [here](https://keras.io/examples/vision/mnist_convnet/).

Inference will be ran in bulk at the end of the day. The output will be displayed directly to end users.

## Potential solution

- SageMaker training and inference pipeline using [Batch Transform](https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html)
    - To trigger inference:
        - Use event bridge to trigger inference on a cron schedule (we know it is as nightly job) - may need to assume location of input data
        - Use API Gateway with Lambda to trigger manually - could have input data (or location of data) as an argument
            - if large quantities, may not want to send via post request, [API Gateway](https://docs.aws.amazon.com/apigateway/latest/developerguide/limits.html) has a total request limit of 10MB and `x_train` (60,000 numbers) has a json size of 160 MB.


    - To trigger training:
        - Use event bridge to trigger inference on a cron schedule (we know it is as nightly job)
        - Use CICD (e.g. github actions) to trigger a new training job
        - Could automatically 'approve' the model so it is available for batch inference

## CICD
Github actions?

## Infra
Terraform

## Resources
- Sagemaker MLOps template [article](https://aws.amazon.com/blogs/machine-learning/build-mlops-workflows-with-amazon-sagemaker-projects-gitlab-and-gitlab-pipelines/) and [repo](https://github.com/aws-samples/sagemaker-custom-project-templates/tree/main/mlops-template-gitlab)

## To run

In AWS account, create bucket `mnist-mlops-v1` in `eu-west-1` region, create a `SAGEMAKER_PIPELINE_ROLE_ARN` with SageMaker IAM rights (like `AmazonSageMakerFullAccess`) and IAM rights granting access to the SageMaker S3 buckets.

### Model build

In your AWS env (check account by running `aws sts get-caller-identity --query Account --output text`)

```bash
cd build
conda create -n mnist-mlops-build -y
conda activate mnist-mlops-build
conda install pip -y
sh run.sh
```

### Model batch run

Generate input data and upload to s3:
```bash
cd deploy
python process_save_data.py
sh data-upload.sh mnist-mlops-v1 input-data mnist.json
```

Run batch transform job:
```bash
export AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
python run-batch-transform-job.py --region eu-west-1 --s3-bucket mnist-mlops-v1 --s3-data-path input-data --model-package-name mnist-mlops-v1 --model-execution-role-arn arn:aws:iam::"${AWS_ACCOUNT}":role/a204384-sagemaker-tpa
```
