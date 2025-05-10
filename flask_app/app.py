import sys
import os
from typing import Dict, Any
from flask import Flask, request, jsonify

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import utility functions after environment setup
from utils import search_articles, concatenate_content, generate_answer, get_or_create_session

app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query() -> Dict[str, Any]:
    """Process search queries and return responses"""
    try:
        # Extract payload
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        user_query = data.get('query')
        session_id = data.get('session_id')

        if not user_query:
            return jsonify({"error": "No query provided"}), 400

        # Get or create session
        session_id = get_or_create_session(session_id)

        # Step 1: Search for articles
        urls = search_articles(user_query)

        # Step 2: Fetch and concatenate content from articles
        all_content = concatenate_content(urls)

        # Step 3: Generate answer using LLM
        answer = generate_answer(all_content, user_query, session_id)

        # Return JSON response
        return jsonify({
            "answer": answer,
            "session_id": session_id,
            "sources_count": len(urls)
        })

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)