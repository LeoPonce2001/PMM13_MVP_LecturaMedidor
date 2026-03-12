import tensorflow as tf
import numpy as np
from PIL import Image

interpreter = tf.lite.Interpreter(model_path="models/dig-class11_1701_s2.tflite")
interpreter.allocate_tensors()


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def predictDigit(img):

    img = img.resize((20, 32))

    img = img.convert("RGB")


    input_data = np.array(img, dtype=np.float32)


    input_data = np.expand_dims(input_data, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])


    arg = np.argmax(output)


    prob = output[0][arg]

    threshold = .90

    if prob < threshold:
        return None, output[0]
    
    results = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, None]
    return results[arg], output[0]
    
