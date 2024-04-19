import os
import yaml
import logging
from box import Box
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from flask_cors import CORS
import timeit
from llm.wrapper import setup_qa_chain, query_embeddings
 
# Importing the allowed_file and run_ingest functions from methods.py
from methods import allowed_file, run_ingest
 
# Initialize Flask app
app = Flask(__name__)
CORS(app)
 
# Define upload folder
UPLOAD_FOLDER = '/home/genaidevassetv1/GenAI/Genesis/data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['genesis']
collection = db['ragdocs']
 
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
# Load configuration from YAML file
with open('config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = Box(yaml.safe_load(ymlfile))
 
# Route for handling file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and store in folder and MongoDB."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request!'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading!'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
            # Insert metadata into MongoDB
            metadata = {
                'document': filename,
                'version': '1.0',
                'date': timeit.default_timer(),
                'status': 'uploaded'
            }
            result = collection.insert_one(metadata)
            # Call run_ingest function to process the uploaded file
            run_ingest(filename, cfg, logger, collection, app)
            return jsonify({'message': 'File uploaded and stored successfully!', 'file_id': str(result.inserted_id)}), 200
        except Exception as e:
            os.remove(file_path)
            return jsonify({'error': f'Error storing file: {str(e)}'}), 500
    else:
        return jsonify({'error': 'File type not allowed!'}), 400
 
# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)