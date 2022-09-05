from tensorflow import keras
import numpy as np
import json
from json import JSONEncoder


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def load_preprocess_save():
    """"""
    (_, _), (x_test, _) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_test = x_test.astype("float32") / 255

    # Make sure images have shape (28, 28, 1)
    # x_test = np.expand_dims(x_test, -1)

    print("x_test shape:", x_test.shape)
    print(x_test.shape[0], "test samples")

    # save data to json in common structure
    # https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-inference.html:
    # let request = {
    #     "instances": [
    #         // First instance.
    #         {
    #         "features": [ 1.5, 16.0, 14.0, 23.0 ]
    #         },
    #         // Second instance.
    #         {
    #         "features": [ -2.0, 100.2, 15.2, 9.2 ]
    #         }
    #     ]
    # }

    print("structure input data")
    input_data = ""
    for x in x_test:
        features = ", ".join(repr(e) for e in x.tolist())
        input_data += f"{{{features}}}\n"

    print("save data to json")
    with open("data/mnist.jsonlines", "w") as write_file:
        write_file.write(input_data)


if __name__ == "__main__":
    load_preprocess_save()
