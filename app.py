import tensorflow as tf
import numpy as np
import cv2
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

# Define the model path (adjust the path if necessary)
model_path = os.path.join(os.path.dirname(__file__), 'models', 'kidney_model.tflite')

# Load the TFLite model from the models folder
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get model input details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Helper function to preprocess the image before passing it to the model
def preprocess_image(image):
    image = image.resize((150, 150))  # Resize the image to the model input size
    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array = image_array / 255.0  # Normalize the image pixels
    return image_array

# Helper function to return suggestions based on the prediction
def get_suggestion(prediction):
    suggestions = {
        "Stone": "It is suggested to consult a doctor for proper diagnosis and treatment.",
        "Healthy": "Great! Keep maintaining a healthy lifestyle with balanced hydration and diet."
    }
    return suggestions.get(prediction, "No suggestion available.")

# Route for image prediction
@app.route('/predict_kidney_stone', methods=['POST'])
def predict_kidney_stone():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        image = Image.open(io.BytesIO(file.read()))
        image_array = preprocess_image(image)
        
        # Set the tensor for inference
        interpreter.set_tensor(input_details[0]['index'], image_array)
        interpreter.invoke()
        
        # Get the model's output
        output = interpreter.get_tensor(output_details[0]['index'])
        prediction = np.argmax(output, axis=1)[0]
        
        # Map prediction to label
        result = 'Stone' if prediction == 1 else 'Healthy'
        
        # Get the suggestion for the prediction
        suggestion = get_suggestion(result)
        
        return jsonify({'prediction': result, 'suggestion': suggestion}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

#
