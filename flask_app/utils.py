import os
import requests
import hashlib
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from typing import List, Optional
from langchain_groq import ChatGroq
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

SESSION_STORAGE = {}

def search_articles(query: str, max_results: int = 5) -> List[str]:
    """
    Search for articles using DuckDuckGo and return just the URLs.
    """
    try:
        # Initialize DuckDuckGo search wrapper directly
        search_wrapper = DuckDuckGoSearchAPIWrapper(
            max_results=max_results,
            region="wt-wt",
            safesearch="moderate",
            time="y",
            backend="auto",
        )

        # Get raw search results as list of dictionaries
        results = search_wrapper.results(f"{query}", max_results=max_results)

        # Extract just the URLs from the results
        urls = [result.get('link') for result in results if result.get('link')]

        return urls

    except Exception as e:
        print(f"Error searching with DuckDuckGo: {e}")
        return []

def fetch_article_content(url: str) -> str:
    """Fetch and extract content from a single URL"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, 'html.parser')

        # Remove script and style elements
        for script in soup(['script', 'style']):
            script.decompose()

        # Get text from paragraph elements
        paragraphs = [p.get_text().strip() for p in soup.find_all(['p', 'h1', 'h2', 'h3'])]
        return '\n\n'.join([p for p in paragraphs if p])
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ''

def concatenate_content(urls: List[str], max_length: int = 8000) -> str:
    """Concatenate content from multiple URLs up to max_length"""
    full_text = []
    current_length = 0

    for url in urls:
        print(f"\nProcessing URL: {url}")

        article_text = fetch_article_content(url)
        if not article_text:
            continue

        source_text = f"\n\nSOURCE: {url}\n{article_text}"

        # Check if adding this would exceed max length
        if current_length + len(source_text) > max_length:
            remaining = max_length - current_length
            if remaining > 100:  # Only add if we can fit a meaningful amount
                full_text.append(source_text[:remaining])
            break

        full_text.append(source_text)
        current_length += len(source_text)

    return '\n'.join(full_text)

def generate_answer(content: str, query: str, session_id: str = None) -> str:
    """Generate answer using LLM API with context from previous interactions"""
    try:
        api_key = os.getenv('GROQ_API_KEY')
        llm = ChatGroq(api_key=api_key, model="gemma2-9b-it", temperature=0.8)

        # Convert session messages to LangChain format if they exist
        langchain_messages = []

        # Add context from session if available
        if session_id and session_id in SESSION_STORAGE:
            for msg in SESSION_STORAGE[session_id]['messages']:
                if msg['role'] == 'system':
                    langchain_messages.append(SystemMessage(content=msg['content']))
                elif msg['role'] == 'user':
                    langchain_messages.append(HumanMessage(content=msg['content']))
                elif msg['role'] == 'assistant':
                    langchain_messages.append(AIMessage(content=msg['content']))

        # If no session or new session, start with system message
        if not langchain_messages:
            langchain_messages = [SystemMessage(content="You are a helpful assistant that provides information based on the given context.")]

        # Add user query with context
        user_message = f"Context:\n{content[:6000]}\n\nUser question: {query}\n\nPlease provide a comprehensive answer based on the context provided."
        langchain_messages.append(HumanMessage(content=user_message))

        # Invoke the model
        response = llm.invoke(langchain_messages)
        answer = response.content

        # Store in session
        if session_id:
            if session_id not in SESSION_STORAGE:
                SESSION_STORAGE[session_id] = {'messages': []}

            # Convert to our simple format for storage
            if not SESSION_STORAGE[session_id]['messages']:
                SESSION_STORAGE[session_id]['messages'].append({
                    'role': 'system',
                    'content': "You are a helpful assistant that provides information based on the given context."
                })

            # Add the query and response to session
            SESSION_STORAGE[session_id]['messages'].append({
                'role': 'user',
                'content': query
            })
            SESSION_STORAGE[session_id]['messages'].append({
                'role': 'assistant',
                'content': answer
            })

            # Keep only last 5 exchanges (10 messages) to manage context length
            if len(SESSION_STORAGE[session_id]['messages']) > 10:
                # Keep system message and recent exchanges
                SESSION_STORAGE[session_id]['messages'] = [SESSION_STORAGE[session_id]['messages'][0]] + SESSION_STORAGE[session_id]['messages'][-9:]

        return answer
    except Exception as e:
        import traceback
        print(f"Error in generate_answer: {e}")
        print(traceback.format_exc())
        return f"Sorry, I encountered an error generating your answer: {str(e)}"

def get_or_create_session(session_id: Optional[str] = None) -> str:
    """Get existing session or create a new one"""
    if not session_id:
        # Generate a random session ID
        session_id = hashlib.md5(os.urandom(16)).hexdigest()

    if session_id not in SESSION_STORAGE:
        SESSION_STORAGE[session_id] = {
            'messages': []
        }

    return session_id