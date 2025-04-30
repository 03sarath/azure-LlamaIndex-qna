from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from llama_index import GPTSimpleVectorIndex, LLMPredictor, LangchainEmbedding
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
import openai

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure Azure OpenAI Service API
openai.api_type = "azure"
openai.api_version = "2023-03-15-preview"
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize LLM and embeddings
llm = AzureChatOpenAI(
    deployment_name="gpt-4o",
    temperature=0,
    openai_api_version="2023-03-15-preview"
)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=1)
llm_predictor = LLMPredictor(llm=llm)
embedding_llm = LangchainEmbedding(embeddings)

# Load the index
index = GPTSimpleVectorIndex.load_from_disk("index.json")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        question = data.get('question')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        # Query the index
        response = index.query(question)
        
        return jsonify({
            'answer': str(response)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 