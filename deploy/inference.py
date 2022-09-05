import json
import numpy as np


def input_handler(data, context):
    """Pre-process request input before it is sent to TensorFlow Serving REST API
    Args:
        data (obj): the request data, in format of dict or string
        context (Context): an object containing request and configuration details
    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """
    print("HERE>>>>")
    print(context.request_content_type)
    if context.request_content_type == "application/jsonlines":
        # pass through json (assumes it's correctly formed)
        decoded_data = data.read().decode("utf-8")
        print(decoded_data)
        numpy_arr = np.array([decoded_data])
        print(numpy_arr)
        return numpy_arr

    raise ValueError('{{"error": "unsupported content type {}"}}'.format(context.request_content_type or "unknown"))


def output_handler(data, context):
    """Post-process TensorFlow Serving output before it is returned to the client.
    Args:
        data (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, response content type
    """
    if data.status_code != 200:
        raise ValueError(data.content.decode("utf-8"))

    response_content_type = context.accept_header
    prediction = data.content
    return prediction, response_content_type
