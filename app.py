from flask import Flask, render_template, request, jsonify
import os
import sys
from dotenv import load_dotenv

# Try to import required packages with better error handling
try:
    from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, LLMPredictor, PromptHelper, LangchainEmbedding
    from langchain.chat_models import AzureChatOpenAI
    from langchain.embeddings import OpenAIEmbeddings
    import openai
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please make sure all dependencies are installed by running:")
    print("pip install -r requirements.txt")
    sys.exit(1)

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Check for required environment variables
required_env_vars = ['OPENAI_API_KEY', 'OPENAI_API_BASE']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
    print("Please create a .env file with the required variables.")
    sys.exit(1)

# Configure Azure OpenAI Service API
openai.api_type = "azure"
openai.api_version = "2023-03-15-preview"
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the index
def initialize_index():
    try:
        print("Starting index initialization...")
        
        # Check if index.json exists
        if os.path.exists("index.json"):
            print("Loading existing index from index.json...")
            index = GPTSimpleVectorIndex.load_from_disk("index.json")
            print("Index loaded successfully from disk")
            return index
        
        # If no index.json exists, create a new index
        print("No existing index found. Creating new index...")
        
        # Check if data directory exists
        data_dir = 'data/qna/'
        if not os.path.exists(data_dir):
            print(f"Error: Data directory '{data_dir}' not found.")
            print("Please create the directory and add your documents.")
            return None
        
        print(f"Found data directory: {data_dir}")
        print("Contents of data directory:", os.listdir(data_dir))

        # Use AzureChatOpenAI for chat-based models
        print("Initializing AzureChatOpenAI...")
        llm = AzureChatOpenAI(
            deployment_id="gpt-35-turbo",
            temperature=0,
            openai_api_base=os.getenv('OPENAI_API_BASE'),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_type="azure",
            openai_api_version="2023-03-15-preview"
        )
        print("AzureChatOpenAI initialized successfully")
        
        print("Creating LLMPredictor...")
        llm_predictor = LLMPredictor(llm=llm)
        print("LLMPredictor created successfully")
        
        print("Initializing embeddings...")
        # Configure OpenAIEmbeddings for Azure
        embedding_llm = LangchainEmbedding(OpenAIEmbeddings(
            model="text-embedding-ada-002",
            deployment_id="text-embedding-ada-002",
            openai_api_base=os.getenv('OPENAI_API_BASE'),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_type="azure",
            openai_api_version="2023-03-15-preview",
            chunk_size=1
        ))
        print("Embeddings initialized successfully")

        # Load documents
        print("Loading documents...")
        documents = SimpleDirectoryReader(data_dir).load_data()
        if not documents:
            print("Error: No documents found in the data directory.")
            return None
        print(f"Successfully loaded {len(documents)} documents")

        # Define prompt helper
        print("Setting up prompt helper...")
        max_input_size = 3000
        num_output = 256
        chunk_size_limit = 1000
        max_chunk_overlap = 20
        prompt_helper = PromptHelper(max_input_size=max_input_size, num_output=num_output, 
                                   max_chunk_overlap=max_chunk_overlap, chunk_size_limit=chunk_size_limit)
        print("Prompt helper configured successfully")

        # Create index
        print("Creating vector index...")
        index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, 
                                   embed_model=embedding_llm, prompt_helper=prompt_helper)
        print("Vector index created successfully")
        
        # Save the index to disk
        print("Saving index to disk...")
        index.save_to_disk("index.json")
        print("Index saved to disk successfully")
        
        return index
    except Exception as e:
        print(f"Error initializing index: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        return None

# Initialize the index
print("Starting application initialization...")
index = initialize_index()
if index is None:
    print("Failed to initialize the index. The application will start but queries will fail.")
else:
    print("Index initialized successfully!")

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
        response = index.query(question)
        return jsonify({'answer': str(response)})
    except Exception as e:
        print(f"Error during query: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 