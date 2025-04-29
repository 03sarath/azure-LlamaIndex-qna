from flask import Flask, render_template, request, jsonify
import os
import logging
from dotenv import load_dotenv
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, LLMPredictor, PromptHelper, LangchainEmbedding
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
import openai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Debug log environment variables (without exposing sensitive data)
logger.info("Verifying environment variables...")
logger.info(f"OPENAI_API_BASE: {os.getenv('OPENAI_API_BASE')}")
logger.info(f"OPENAI_API_VERSION: {os.getenv('OPENAI_API_VERSION')}")
logger.info(f"OPENAI_API_TYPE: {os.getenv('OPENAI_API_TYPE')}")
logger.info(f"OPENAI_DEPLOYMENT_NAME: {os.getenv('OPENAI_DEPLOYMENT_NAME')}")
logger.info("OPENAI_API_KEY: [REDACTED]")

# Validate required environment variables
required_env_vars = [
    'OPENAI_API_BASE',
    'OPENAI_API_KEY',
    'OPENAI_API_VERSION',
    'OPENAI_API_TYPE',
    'OPENAI_DEPLOYMENT_NAME'
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Configure Azure OpenAI environment variables
os.environ["OPENAI_API_TYPE"] = os.getenv('OPENAI_API_TYPE')
os.environ["OPENAI_API_VERSION"] = os.getenv('OPENAI_API_VERSION')
os.environ["OPENAI_API_BASE"] = os.getenv('OPENAI_API_BASE')
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Configure OpenAI client
openai.api_type = os.getenv('OPENAI_API_TYPE')
openai.api_version = os.getenv('OPENAI_API_VERSION')
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize the index
def initialize_index():
    try:
        logger.info("Starting index initialization...")
        
        # Check if index.json exists
        if os.path.exists("index.json"):
            logger.info("Loading existing index from index.json...")
            index = GPTSimpleVectorIndex.load_from_disk("index.json")
            logger.info("Index loaded successfully from disk")
            return index
        
        # If no index.json exists, create a new index
        logger.info("No existing index found. Creating new index...")
        
        # Check if data directory exists
        data_dir = 'data/qna/'
        if not os.path.exists(data_dir):
            logger.error(f"Error: Data directory '{data_dir}' not found.")
            logger.error("Please create the directory and add your documents.")
            return None
        
        logger.info(f"Found data directory: {data_dir}")
        logger.info(f"Contents of data directory: {os.listdir(data_dir)}")

        # Initialize AzureChatOpenAI
        deployment_name = os.getenv('OPENAI_DEPLOYMENT_NAME')
        logger.info(f"Initializing AzureChatOpenAI with deployment: {deployment_name}")
        
        llm = AzureChatOpenAI(
            deployment_name=deployment_name,
            temperature=0,
            openai_api_version=os.getenv('OPENAI_API_VERSION')
        )
        logger.info("AzureChatOpenAI initialized successfully")
        
        # Create LLMPredictor
        logger.info("Creating LLMPredictor...")
        llm_predictor = LLMPredictor(llm=llm)
        logger.info("LLMPredictor created successfully")
        
        # Initialize embeddings
        logger.info("Initializing embeddings...")
        embedding_llm = LangchainEmbedding(OpenAIEmbeddings(
            model="text-embedding-ada-002",
            engine="text-embedding-ada-002",
            openai_api_base=os.getenv('OPENAI_API_BASE'),
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            openai_api_type="azure",
            openai_api_version=os.getenv('OPENAI_API_VERSION'),
            chunk_size=1
        ))
        logger.info("Embeddings initialized successfully")

        # Load documents
        logger.info("Loading documents...")
        documents = SimpleDirectoryReader(data_dir).load_data()
        if not documents:
            logger.error("Error: No documents found in the data directory.")
            return None
        logger.info(f"Successfully loaded {len(documents)} documents")

        # Define prompt helper
        logger.info("Setting up prompt helper...")
        max_input_size = 3000
        num_output = 256
        chunk_size_limit = 1000
        max_chunk_overlap = 20
        prompt_helper = PromptHelper(
            max_input_size=max_input_size,
            num_output=num_output,
            max_chunk_overlap=max_chunk_overlap,
            chunk_size_limit=chunk_size_limit
        )
        logger.info("Prompt helper configured successfully")

        # Create index
        logger.info("Creating vector index...")
        index = GPTSimpleVectorIndex(
            documents,
            llm_predictor=llm_predictor,
            embed_model=embedding_llm,
            prompt_helper=prompt_helper
        )
        logger.info("Vector index created successfully")
        
        # Save the index to disk
        logger.info("Saving index to disk...")
        index.save_to_disk("index.json")
        logger.info("Index saved to disk successfully")
        
        return index
    except Exception as e:
        logger.error(f"Error initializing index: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error("Full traceback:")
        traceback.print_exc()
        return None

# Initialize the index
logger.info("Starting application initialization...")
index = initialize_index()
if index is None:
    logger.error("Failed to initialize the index. The application will start but queries will fail.")
else:
    logger.info("Index initialized successfully!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    if not index:
        return jsonify({'error': 'Index not initialized. Please check the server logs for details.'}), 500
    
    data = request.get_json()
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        logger.info(f"Processing question: {question}")
        response = index.query(question)
        logger.info("Question processed successfully")
        return jsonify({'answer': str(response)})
    except Exception as e:
        error_details = {
            'error_message': str(e),
            'error_type': type(e).__name__,
            'error_args': e.args if hasattr(e, 'args') else None,
            'error_dict': e.__dict__ if hasattr(e, '__dict__') else None
        }
        logger.error(f"Error during query: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error args: {e.args if hasattr(e, 'args') else None}")
        logger.error(f"Error dict: {e.__dict__ if hasattr(e, '__dict__') else None}")
        import traceback
        logger.error("Full traceback:")
        traceback.print_exc()
        return jsonify({'error': str(e), 'details': error_details}), 500

if __name__ == '__main__':
    app.run(debug=True) 