import os
import yaml
import logging
from box import Box
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, UnstructuredWordDocumentLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredExcelLoader
 
def allowed_file(filename):
    """Check if the file extension is allowed."""
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'doc', 'xlsx'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 
def process_text_documents(documents, cfg, logger):
    """Process text documents."""
    logger.info("Processing text documents...")
    # Split text documents into smaller chunks for processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.CHUNK_SIZE, chunk_overlap=cfg.CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)
 
    # Generate embeddings for the text using a pre-trained model
    embeddings = HuggingFaceEmbeddings(model_name=cfg.EMBEDDINGS, model_kwargs={'device': 'cpu'})
    vectorstore = FAISS.from_documents(texts, embeddings)
 
    # Save the embeddings to a local database for future retrieval
    vectorstore.save_local(cfg.DB_FAISS_PATH)
    logger.info("Text documents processing completed.")
 
def run_ingest(filename, cfg, logger, collection, app):
    """Handle ingestion and chunking of the uploaded file."""
    logger.info(f"Starting ingestion process for file: {filename}")
    supported_document_types = set(cfg.DOCUMENT_TYPE)
    file_extension = filename.split('.')[-1].lower()
 
    # Determine the appropriate loader based on the file extension
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
 
    # Load the document using the appropriate loader
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    loader = loader_cls(file_path)
    documents = loader.load()
 
    # Process the documents based on their type
    if file_extension == 'txt':
        process_text_documents(documents, cfg, logger)  # Process text documents separately
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
 
    # Update the metadata status to 'ingested' after processing
    try:
        collection.update_one({'document': filename}, {'$set': {'status': 'ingested'}})
        logger.info(f"Status updated to 'ingested' for file: {filename}")
    except Exception as e:
        logger.error(f"Error updating status for file {filename}: {e}")
    logger.info(f"Ingestion process completed for file: {filename}")