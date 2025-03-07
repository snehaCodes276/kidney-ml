from flask import Flask, request, jsonify
import numpy as np
import cv2
import os
import tensorflow as tf

app = Flask(__name__)

# Load the TFLite model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'kidney_model.tflite')
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Labels (must match training order)
labels = ["Normal", "Stone"]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['file']
    try:
        # Read and preprocess the image
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (150, 150))  # Resize to match model input (150x150)
        image = image / 255.0  # Normalize
        image = np.expand_dims(image, axis=0).astype(np.float32)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]

        # Get prediction and confidence
        predicted_class = labels[np.argmax(output)]
        confidence = float(np.max(output))

        return jsonify({
            "prediction": predicted_class,
            "confidence": confidence,
            "suggestion": get_suggestion(predicted_class)
        }), 200

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)[:-1]}"}), 500

def get_suggestion(prediction):
    if prediction == "Stone":
        return (
            "A kidney stone is detected. Please consult a urologist immediately. "
            "Increase your water intake to help pass small stones naturally. "
            "Avoid high-oxalate foods like spinach and nuts. "
            "Pain relievers may help, but medical guidance is recommended."
        )
    else:
        return (
            "Your kidney appears healthy. Maintain a balanced diet, "
            "stay hydrated, and get regular checkups for optimal health."
        )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
