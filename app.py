import os
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Set model path and load your TFLite model
MODEL_DIR = 'models/'
MODEL_PATH = os.path.join(MODEL_DIR, 'kidney_model.tflite')
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Helper function to preprocess the image and make predictions
def preprocess_image(image):
    # Resize image to the model's expected input size (150x150)
    image = image.resize((150, 150))
    image_array = np.array(image)
    # Normalize and expand dimensions for the batch
    image_array = image_array.astype(np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Define prediction function
def predict(image):
    image_array = preprocess_image(image)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], image_array)
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensor
    output = interpreter.get_tensor(output_details[0]['index'])
    
    # Output prediction (Assuming output is categorical)
    prediction = np.argmax(output, axis=1)
    return prediction[0]

# Flask route to handle image upload and prediction
@app.route('/predict_kidney_stone', methods=['POST'])
def upload_and_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Read image from the file
        img = Image.open(file.stream)
        
        # Make prediction
        prediction = predict(img)
        
        # Provide a suggestion based on the prediction
        if prediction == 0:
            suggestion = "The image seems to represent a 'normal' condition. This is typically the case for healthy kidney images."
        elif prediction == 1:
            suggestion = "The image seems to represent a 'stone' condition. This may indicate the presence of kidney stones. It is advisable to consult a doctor for further diagnosis."
        else:
            suggestion = "Unable to determine the condition. Please ensure the image is clear and correctly labeled."
        
        # Return prediction and suggestion as a response
        return jsonify({
            'prediction': prediction,
            'suggestion': suggestion
        })

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
