import os
from dotenv import load_dotenv
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, LLMPredictor, PromptHelper, LangchainEmbedding
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
import openai

# Load environment variables
load_dotenv()

# Configure Azure OpenAI Service API
openai.api_type = "azure"
openai.api_version = "2023-03-15-preview"
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv("OPENAI_API_KEY")

def create_index():
    print("Starting index creation...")
    
    # Check if data directory exists
    data_dir = 'data/qna/'
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        print("Please create the directory and add your documents.")
        return False
    
    print(f"Found data directory: {data_dir}")
    print("Contents of data directory:", os.listdir(data_dir))

    # Use AzureChatOpenAI for chat-based models
    print("Initializing AzureChatOpenAI...")
    llm = AzureChatOpenAI(
        deployment_id="gpt-35-turbo",
        temperature=0,
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
        chunk_size=1
    ))
    print("Embeddings initialized successfully")

    # Load documents
    print("Loading documents...")
    documents = SimpleDirectoryReader(data_dir).load_data()
    if not documents:
        print("Error: No documents found in the data directory.")
        return False
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
    return True

if __name__ == "__main__":
    success = create_index()
    if success:
        print("Index creation completed successfully!")
    else:
        print("Index creation failed. Please check the error messages above.") 