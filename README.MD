# Web Search Assistant 🔍

A powerful web search application that retrieves relevant information from the internet and generates comprehensive answers using LLMs (Language Model Models).

## 🌟 Features

- **Real-time web search** via DuckDuckGo API
- **Contextual AI responses** using Groq's Gemma 2-9B model
- **Session management** for continuous conversations
- **User-friendly interface** built with Streamlit
- **API backend** powered by Flask
- **Memory management** to maintain conversation context

## 🏗️ Architecture

The application consists of three main components:

1. **Utility Module** (`utils.py`): Core functionality for web searches, content extraction, and LLM integration
2. **Flask Backend** (`flask_app/app.py`): REST API to handle requests and responses
3. **Streamlit Frontend** (`streamlit_app/app.py`): User interface for interacting with the application

```
query-system
├── .env                      # Environment variables
├── requirements.txt
├── .gitignore
├── flask_app/
│   └── app.py                # Core utility functions
│   └── utils.py              # Flask API backend
└── streamlit_app/
│   └── app.py                # Streamlit frontend interface
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- [Groq API key](https://console.groq.com/)
- Internet connection

### Installation

1. Clone this repository:
```bash
git clone https://github.com/santusahoo/query-system.git
cd query-system
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```
GROQ_API_KEY=your_groq_api_key_here
FLASK_API_URL=http://localhost:5001
```

### Running the Application

1. Start the Flask backend:
```bash
cd flask_app
python app.py
```

2. In a new terminal, start the Streamlit frontend:
```bash
cd streamlit_app
streamlit run app.py
```

3. Open your browser and navigate to the URL shown in the Streamlit terminal (typically `http://localhost:8501`)

## 🔍 How It Works

1. User enters a query in the Streamlit interface
2. The query is sent to the Flask backend
3. The backend searches the web using DuckDuckGo
4. Content is extracted from search results using BeautifulSoup
5. The LLM generates a comprehensive answer based on the extracted content
6. The answer is displayed to the user in the Streamlit interface
7. The conversation is saved in the session for context maintenance

## 🔧 Configuration

You can modify the following parameters in the application:

### Search Parameters
- `max_results`: Number of search results to retrieve (default: 5)
- `max_length`: Maximum length of concatenated content (default: 8000)

### LLM Parameters
- `model`: LLM model to use (default: "gemma2-9b-it")
- `temperature`: Creativity level of responses (default: 0.8)

## 📊 Performance Considerations

- Web content extraction can be time-consuming for complex websites
- LLM inference speed depends on the Groq API response time
- Session storage is currently in-memory and not persistent across restarts


## 🙏 Acknowledgements

- [DuckDuckGo](https://duckduckgo.com/) for search capabilities
- [Groq](https://groq.com/) for LLM inference
- [Streamlit](https://streamlit.io/) for the frontend framework
- [Flask](https://flask.palletsprojects.com/) for the API backend
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) for web scraping
- [LangChain](https://www.langchain.com/) for LLM integration

---

Made with ❤️ by Santu