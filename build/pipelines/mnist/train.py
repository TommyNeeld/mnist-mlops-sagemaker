import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import logging
import argparse
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def _fetch_data():
    # Load the data and split it between train and test sets
    logger.info("Loading data.")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    # TODO: this should be done in a preprocessing step
    logger.info("Starting preprocessing.")
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    return x_train, y_train, x_test, y_test


def train(args):

    # Model / data parameters
    num_classes = 10
    input_shape = (28, 28, 1)

    # gets data
    x_train, y_train, x_test, y_test = _fetch_data()

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # train model
    logger.info("Create model.")
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    batch_size = args.batch_size
    epochs = args.epochs

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    logger.info("Train model.")
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=2)

    logger.info("Save model.")

    # Save the model
    # A version number is needed for the serving container
    # to load the model
    version = "00000000"
    ckpt_dir = os.path.join(args.model_dir, version)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    model.save(ckpt_dir)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)

    # Environment variables given by the training image
    parser.add_argument("--model-dir", type=str, default=os.getenv("SM_MODEL_DIR"))
    parser.add_argument("--output-dir", type=str, default=os.getenv("SM_OUTPUT_DIR"))
    parser.add_argument("--train", type=str, default=os.getenv("SM_CHANNEL_TRAINING"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"args: {args}")
    train(args)
