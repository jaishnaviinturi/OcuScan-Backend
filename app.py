import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import gdown
import logging
from tensorflow.lite.python.interpreter import Interpreter as tflite  # Corrected import for TFLite interpreter
from tensorflow.keras.preprocessing.image import img_to_array

# Set environment variable to disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend compatibility

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define class labels based on the dataset
class_labels = ['cataract', 'normal fundus', 'pathological myopia', 'moderate non proliferative retinopathy', 
                'dry age-related macular degeneration', 'glaucoma', 'mild nonproliferative retinopathy']

# Model path and Google Drive URL
model_path = os.getenv("MODEL_PATH", "model_dynamic_quantized.tflite")
model_url = os.getenv("MODEL_URL", "https://drive.google.com/uc?id=1XwDeaiq05C1BcnYDTIEskKNQQJmsmJyA")

# Ensure the output directory exists
os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)

# Download model if it doesn't exist
if not os.path.exists(model_path):
    logger.info("Downloading model file from Google Drive...")
    try:
        gdown.download(model_url, model_path, quiet=False)
        logger.info("Model downloaded successfully")
    except Exception as e:
        logger.error(f"Failed to download model: {str(e)}")
        raise

# Load the TensorFlow Lite model
try:
    interpreter = tflite(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logger.info("TensorFlow Lite model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

def preprocess_image(image_file, target_size=(128, 128)):
    """
    Preprocess an uploaded image for the TFLite model.
    
    Args:
        image_file: Uploaded image file
        target_size (tuple): Target size for resizing the image (128x128)
    
    Returns:
        numpy array: Preprocessed image
    """
    try:
        img_array = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = cv2.resize(img, target_size)  # Resize to model input size
        img = img_to_array(img) / 255.0  # Normalize to [0, 1]
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = img.astype(np.float32)  # Ensure float32 for dynamic quantization
        return img
    except Exception as e:
        logger.error(f"Image preprocessing error: {str(e)}")
        raise

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict ocular disease from an uploaded image.
    
    Returns:
        JSON: Predicted class only
    """
    logger.info("Received prediction request")
    try:
        if 'image' not in request.files:
            logger.error("No image file provided")
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        processed_image = preprocess_image(image_file)
        
        # Run inference with TensorFlow Lite
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        predicted_class_idx = np.argmax(predictions)
        predicted_class = class_labels[predicted_class_idx]
        
        logger.info(f"Prediction successful: {predicted_class}")
        response = {'predicted_class': predicted_class}
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)), debug=False)