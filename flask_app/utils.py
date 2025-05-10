import os
import requests
import hashlib
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional

load_dotenv()

SESSION_STORAGE = {}

# Using Serper API for search
def search_articles(query: str) -> List[Dict[str, str]]:
    """Search for articles using Serper API"""
    api_key = os.getenv('SERPER_API_KEY')
    if not api_key:
        raise ValueError("SERPER_API_KEY environment variable not set")

    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }

    payload = {
        'q': query,
        'gl': 'us',
        'num': 3,
    }

    try:
        response = requests.post(
            'https://google.serper.dev/search',
            headers=headers,
            json=payload
        )
        response.raise_for_status()

        search_results = response.json()
        organic_results = search_results.get('organic', [])

        articles = []
        for result in organic_results:
            articles.append({
                'title': result.get('title', ''),
                'snippet': result.get('snippet', ''),
                'url': result.get('link', '')
            })

        return articles
    except Exception as e:
        print(f"Error in search_articles: {e}")
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

def concatenate_content(articles: List[Dict[str, str]], max_length: int = 8000) -> str:
    """Concatenate content from multiple articles up to max_length"""
    full_text = []
    current_length = 0

    for article in articles:
        url = article.get('url')
        print("\n")
        print(f"Processing URL: {url}")
        if not url:
            continue

        article_text = fetch_article_content(url)
        if not article_text:
            continue

        source_text = f"\n\nSOURCE: {url}\n{article_text}"

        # Check if adding this would exceed max length
        if current_length + len(source_text) > max_length:
            remaining = max_length - current_length
            if remaining > 1000:  # Only add if we can fit a meaningful amount
                full_text.append(source_text[:remaining])
            break

        full_text.append(source_text)
        current_length += len(source_text)

    return '\n'.join(full_text)


def generate_answer(content: str, query: str, session_id: str = None) -> str:
    """Generate answer using OpenAI API with context from previous interactions"""
    try:
        # Create OpenAI client properly
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        messages = []

        # Add context from session if available
        if session_id and session_id in SESSION_STORAGE:
            messages = SESSION_STORAGE[session_id]['messages']

        # If no session or new session, start with system message
        if not messages:
            messages = [{
                'role': 'system',
                'content': 'You are a helpful assistant that provides information based on the given context.'
            }]

        # Add user query with context
        messages.append({
            'role': 'user',
            'content': f"Context:\n{content}\n\nUser question: {query}\n\nPlease provide a comprehensive answer based on the context provided."
        })

        # Limit context length to avoid token limits
        if sum(len(msg['content']) for msg in messages) > 16000:
            # Keep system message and at most 2 previous exchanges plus current query
            if len(messages) > 4:
                messages = [messages[0]] + messages[-4:]

        response = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=messages,
            temperature=0.6,
            max_tokens=1000
        )

        answer = response.choices[0].message.content.strip()

        # Store in session
        if session_id:
            if session_id not in SESSION_STORAGE:
                SESSION_STORAGE[session_id] = {'messages': []}

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
                SESSION_STORAGE[session_id]['messages'] = SESSION_STORAGE[session_id]['messages'][-10:]

        return answer
    except Exception as e:
        print(f"Error in generate_answer: {e}")
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