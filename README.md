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

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
