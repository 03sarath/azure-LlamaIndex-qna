# Azure LlamaIndex Q&A System

This project demonstrates how to build a Question & Answer system using LlamaIndex (formerly GPT Index) with Azure OpenAI Service.

## 🚀 Quick Start

### 1. Setup Environment

1. **Activate your virtual environment:**
   ```bash
  python3.11 -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  pip install -r requirements.txt

   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### 2. Configure Azure OpenAI

1. **Create a `.env` file** in the `qna-quickstart-with-gpt-index/` directory:
   ```
   AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
   AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
   ```

2. **Update the model name** in the script to match your Azure deployment:
   - Change `model="gpt-4o"` to your deployed model name (e.g., `"gpt-35-turbo"`, `"gpt-4"`, etc.)


