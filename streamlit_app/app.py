import os
import streamlit as st
import requests
import uuid
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title='Web Search',
    page_icon='🔍',
    layout='wide'
)

def get_session_id():
    """Get or create a session ID"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def initialize_history():
    """Initialize chat history if it doesn't exist"""
    if 'history' not in st.session_state:
        st.session_state.history = []

def add_to_history(query, answer):
    """Add a query-answer pair to the chat history"""
    st.session_state.history.append({
        "query": query,
        "answer": answer,
        "timestamp": time.strftime("%H:%M:%S")
    })

def display_history():
    """Display the chat history"""
    for i, item in enumerate(st.session_state.history):
        # User query
        st.write(f"**You ({item['timestamp']})**: {item['query']}")

        # Assistant answer
        st.write(f"**Assistant**: {item['answer']}")

        # Add separator except for the last item
        if i < len(st.session_state.history) - 1:
            st.markdown("---")

# Initialize
initialize_history()
session_id = get_session_id()

# App title and description
st.title('🔍 Web Search Assistant')
st.markdown("""
    Ask questions and get answers.
    The system searches the web for relevant information and generates answers using LLMs.
""")

# Sidebar with session information
with st.sidebar:
    st.subheader("Session Information")
    st.write(f"Session ID: `{session_id}`")

    # Option to create a new session
    if st.button("New Session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()

    # Advanced settings
    st.subheader("Settings")
    api_url = st.text_input(
        "API URL",
        value=os.getenv('FLASK_API_URL', 'http://localhost:5001'),
        help="URL of the Flask backend API"
    )

    # Clear history button
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

# User input
query = st.text_area('Enter your question:', height=100)

# Search button
col1, col2 = st.columns([1, 5])
with col1:
    search_clicked = st.button('Search', type="primary", use_container_width=True)

# Process search
if search_clicked and query:
    with st.spinner('Searching and generating answer...'):
        try:
            # Make request to backend
            resp = requests.post(
                f"{api_url}/query",
                json={
                    'query': query,
                    'session_id': session_id
                },
                timeout=60
            )

            # Process response
            if resp.status_code == 200:
                result = resp.json()
                answer = result.get('answer', '')
                sources_count = result.get('sources_count', 0)

                # Display answer
                st.markdown("### Answer")
                st.write(answer)

                # Add to history
                add_to_history(query, answer)

                # Display source information
                st.caption(f"Based on {sources_count} sources")

            else:
                st.error(f"Error {resp.status_code}: {resp.text}")

        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {str(e)}")
            st.info("Make sure the Flask backend is running and accessible.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Display chat history
if st.session_state.history:
    st.markdown("### Chat History")
    display_history()

# Footer
st.markdown("---")
st.caption("Search Assistant powered by OpenAI and Serper API")