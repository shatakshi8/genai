import os
import yaml
import logging
from box import Box
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, UnstructuredWordDocumentLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredExcelLoader
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from bson import ObjectId
from flask_cors import CORS
import timeit
from llm.wrapper import setup_qa_chain, query_embeddings

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = '/home/genaidevassetv1/GenAI/Genesis/data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
client = MongoClient('mongodb://localhost:27017/')
db = client['genesis']
collection = db['ragdocs']

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open('config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = Box(yaml.safe_load(ymlfile))

def allowed_file(filename):
   """Check if the file extension is allowed."""
   ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'doc', 'xlsx'}
   return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to process text documents
def process_text_documents(documents):
    logger.info("Processing text documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.CHUNK_SIZE, chunk_overlap=cfg.CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=cfg.EMBEDDINGS, model_kwargs={'device': 'cpu'})
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(cfg.DB_FAISS_PATH)
    logger.info("Text documents processing completed.")

def run_ingest(filename):
   """Handle ingestion and chunking of the uploaded file."""
   logger = logging.getLogger(__name__)
   logger.info(f"Starting ingestion process for file: {filename}")
   supported_document_types = set(cfg.DOCUMENT_TYPE)
   file_extension = filename.split('.')[-1].lower()
   if file_extension == 'pdf' and 'pdf' in supported_document_types:
       loader_cls = PyPDFLoader
   elif file_extension == 'txt' and 'txt' in supported_document_types:
       loader_cls = DirectoryLoader  # Use DirectoryLoader for text files
   elif file_extension == 'docx' and 'docx' in supported_document_types:
       loader_cls = UnstructuredWordDocumentLoader
   elif file_extension == 'xlsx' and 'xlsx' in supported_document_types:
       loader_cls = UnstructuredExcelLoader
   else:
       logger.warning(f"Unsupported document type for file {filename}.")
       return

   if file_extension == 'txt' and 'txt' in supported_document_types:
      file_path = os.path.join(app.config['UPLOAD_FOLDER'])
      loader = loader_cls(file_path, glob=str(filename))

   file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
   loader = loader_cls(file_path)
   documents = loader.load()

   if file_extension == 'txt':
       process_text_documents(documents)  # Process text documents separately
   else:
       text_splitter = RecursiveCharacterTextSplitter(
           chunk_size=cfg.CHUNK_SIZE,
           chunk_overlap=cfg.CHUNK_OVERLAP
     )
       texts = text_splitter.split_documents(documents)
       embeddings = HuggingFaceEmbeddings(
           model_name=cfg.EMBEDDINGS,
           model_kwargs={'device': 'cpu'}
       )
       vectorstore = FAISS.from_documents(texts, embeddings)
       vectorstore.save_local(cfg.DB_FAISS_PATH)
   try:
       # Update metadata status to 'ingested'
       collection.update_one({'document': filename}, {'$set': {'status': 'ingested'}})
       logger.info(f"Status updated to 'ingested' for file: {filename}")
   except Exception as e:
       logger.error(f"Error updating status for file {filename}: {e}")
   logger.info(f"Ingestion process completed for file: {filename}")

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
           metadata = {
               'document': filename,
               'version': '1.0',
               'date': timeit.default_timer(),
               'status': 'uploaded'
           }
           result = collection.insert_one(metadata)
           return jsonify({'message': 'File uploaded and stored successfully!', 'file_id': str(result.inserted_id)}), 200
       except Exception as e:
           os.remove(file_path)
           return jsonify({'error': f'Error storing file: {str(e)}'}), 500
   else:
       return jsonify({'error': 'File type not allowed!'}), 400

@app.route('/files', methods=['GET'])
def get_files():
   """Retrieve files metadata from MongoDB and return as JSON."""
   try:
       metadata = list(collection.find({}, {'document': 1, 'version': 1, 'date': 1, 'status': 1}))
       for data in metadata:
           data['_id'] = str(data['_id'])
       return jsonify(metadata), 200
   except Exception as e:
       return jsonify({'error': f'Internal Server Error: {str(e)}'}), 500

@app.route('/ingest/<fileName>', methods=['POST'])
def ingest_file(fileName):
   """Handle ingestion and chunking of the uploaded file."""
   try:
       run_ingest(fileName)
       return jsonify({'message': 'File ingestion and chunking started successfully!'}), 200
   except Exception as e:
       return jsonify({'error': f'Error ingesting file: {str(e)}'}), 500


## Ask wih input param
@app.route('/ask/<genask>', methods=['POST'])
def asktype(genask):
   mainPrompt = ""
    ##mainPrompt = "Act as a QA engineer, for the given the text" + "'" + question +"'" + ", generate all possible test cases. For the same test case, provi>
   question = request.json['question']

   if genask == "testcases":
        mainPrompt = "Act as a QA engineer, for the given the text " + "'" + question +"'" + ", generate all possible functional and non-functional test cas>"

   elif genask == "teststeps":
        mainPrompt = "Act as a QA engineer, for the given the test cases " + "'" + question +"'" + ", generate elaborated test steps for each input test cas>"

   elif genask == "testscript":
        language = request.json['language']
        mainPrompt = "Act as QA engineer, for the given test case with test steps " + "'" + question + "', generate test script in langage '" + language + ">"
   semantic_search = request.json.get('semantic_search', False)
   start = timeit.default_timer()
   if semantic_search:
      semantic_search_results = query_embeddings(mainPrompt)
      answer = {'semantic_search_results' : semantic_search_results}
   else:
      qa_chain = setup_qa_chain()
      response = qa_chain({'query' : mainPrompt})
      answer = {'answer' : response['result']}

   return jsonify(answer)


@app.route('/ask', methods=['POST'])
def ask():
   """ Ask a question about the invoice data and return the answer """
   question = request.json['question']
   semantic_search = request.json.get('semantic_search', False)
   start = timeit.default_timer()
   if semantic_search:
      semantic_search_results = query_embeddings(question)
      answer = {'semantic_search_results' : semantic_search_results}
   else:
      qa_chain = setup_qa_chain()
      response = qa_chain({'query' : question})
      answer = {'answer' : response['result']}

   end = timeit.default_timer()
   answer['time_taken'] = end - start
   return jsonify(answer)

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """Handle file download."""
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            response = send_file(file_path, as_attachment=True)
            response.headers['Content-Type'] = 'application/octet-stream'
            return response
            #send_file(file_path, as_attachment=True, mimetype = 'application/octet-stream')
        else:
            return jsonify({'error': 'File not found!'}), 404
    except Exception as e:
        return jsonify({'error': f'Error downloading file: {str(e)}'}), 500


if __name__ == '__main__':
   app.run(debug=True, host="0.0.0.0", port=5000)


