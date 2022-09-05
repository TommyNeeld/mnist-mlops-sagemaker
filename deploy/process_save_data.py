from tensorflow import keras
import numpy as np
import boto3
import logging
import argparse
from sagemaker.serializers import JSONLinesSerializer

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def load_preprocess_save(args):
    """load some example batch data, save to jsonlines and upload to s3

    Args:
        args (dict): parsed arguments
    """
    local_filepath = "data/mnist.jsonl"

    (_, _), (x_test, _) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_test = x_test.astype("float32") / 255

    # Make sure images have shape (28, 28, 1)
    x_test = np.expand_dims(x_test, -1)

    print("x_test shape:", x_test.shape)
    print(x_test.shape[0], "test samples")

    print("structure input data")

    serialized_jsonlines = JSONLinesSerializer(content_type="application/jsonlines").serialize(x_test.tolist())
    print("save data to json")
    with open(local_filepath, "w") as write_file:
        write_file.write(serialized_jsonlines)

    # Then upload the resulting datafile to an S3 path.
    print("load to s3")
    s3 = boto3.resource("s3")
    s3.meta.client.upload_file(local_filepath, args.s3_bucket, f"{args.s3_data_path}/mnist.jsonl")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-bucket", type=str, default=None)
    parser.add_argument("--s3-data-path", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"args: {args}")
    load_preprocess_save(args)
