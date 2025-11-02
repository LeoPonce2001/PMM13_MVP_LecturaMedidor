import tensorflow as tf
import numpy as np
from PIL import Image

interpreter = tf.lite.Interpreter(model_path="models/dig-class11_1701_s2.tflite")
interpreter.allocate_tensors()

# Get tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def predictDigit(img):
    # Resize to model input size (width=20, height=32)
    img = img.resize((20, 32))

    # Convert to RGB if not already
    img = img.convert("RGB")

    # Convert to NumPy array and scale to float32
    input_data = np.array(img, dtype=np.float32)

    # Add batch dimension: [1, 32, 20, 3]
    input_data = np.expand_dims(input_data, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    #print(output)
    arg = np.argmax(output)
    #print(f"Predicted: {arg - 1} with prob: {output[0][arg]:.2f}")

    prob = output[0][arg]

    threshold = .90

    if prob < threshold:
        return None, output[0]
    
    results = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, None]
    return results[arg], output[0]
    
