# Azure OpenAI QnA System

A Flask-based web application that uses Azure OpenAI Service and LlamaIndex to provide question-answering capabilities.

## Prerequisites

- Python 3.8 or higher
- Azure OpenAI Service account
- `.env` file with the following variables:
  ```
  OPENAI_API_KEY=your_api_key
  OPENAI_API_BASE=your_azure_openai_endpoint
  ```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Make sure your `.env` file is properly configured with your Azure OpenAI credentials.

2. Start the Flask application:
   ```bash
   python app.py
   ```

3. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

## Features

- Modern, responsive UI built with Bootstrap
- Real-time question answering using Azure OpenAI Service
- Error handling and loading states
- Clean and intuitive user interface

## Project Structure

```
.
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (not in repo)
├── static/
│   ├── style.css      # Custom CSS styles
│   └── script.js      # Frontend JavaScript
├── templates/
│   └── index.html     # Main HTML template
└── index.json         # LlamaIndex data file
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 