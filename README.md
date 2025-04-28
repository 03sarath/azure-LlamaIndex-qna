# Azure OpenAI Q&A System with LlamaIndex

This is a Flask-based web application that provides a user interface for querying documents using Azure OpenAI Service and LlamaIndex.

## Features

- Web interface for asking questions about your documents
- Uses Azure OpenAI Service for embeddings and chat
- Built with LlamaIndex for efficient document retrieval
- Modern and responsive UI

## Prerequisites

- Python 3.8 or higher
- Azure OpenAI Service account
- Documents to be indexed (place them in the `data/qna/` directory)

## Setup

1. Create a `.env` file in the root directory with the following content:
```
OPENAI_API_KEY=your_azure_openai_api_key
OPENAI_API_BASE=your_azure_openai_endpoint
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Place your documents in the `data/qna/` directory. The application will index these documents.

## Running the Application

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Enter your question in the text area and click "Ask Question" to get answers from your documents.

## Project Structure

- `app.py`: Main Flask application
- `templates/index.html`: Web interface template
- `data/qna/`: Directory containing documents to be indexed
- `requirements.txt`: Python dependencies
- `.env`: Environment variables (not tracked in git)

## Notes

- The application uses Azure OpenAI Service for both embeddings and chat functionality
- Documents are indexed when the application starts
- The index is stored in memory and will need to be rebuilt if the application restarts 