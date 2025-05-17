import os
import streamlit as st
import time
import logging
from typing import Optional
from dotenv import load_dotenv

from rag_app import RAGApplication

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Document Chatbot",
    page_icon="📚",
    layout="wide"
)

# Apply custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stTextInput>div>div>input {
        border-radius: 20px;
    }
    .stButton>button {
        border-radius: 20px;
        width: 100%;
    }
    .st-emotion-cache-16txtl3 p {
        word-break: break-word;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_rag_application():
    """Initialize and return RAG application."""
    app = RAGApplication()
    return app

def main():
    # Header
    st.title("📚 Document Q&A Chatbot")
    st.markdown("Ask questions about your company documents and get instant answers.")
    
    # Sidebar
    st.sidebar.title("Options")
    
    # Document ingestion section in sidebar
    st.sidebar.header("Document Management")
    
    data_dir = st.sidebar.text_input("Document Directory Path", "data")
    
    if st.sidebar.button("Ingest Documents"):
        with st.spinner("Ingesting documents... This may take a while"):
            app = get_rag_application()
            success = app.ingest_documents(directory_path=data_dir)
            
            if success:
                st.sidebar.success("Documents ingested successfully!")
            else:
                st.sidebar.error("Failed to ingest documents. Check logs for details.")
    
    # Advanced options in sidebar
    st.sidebar.header("Search Options")
    k = st.sidebar.slider("Number of documents to retrieve", 1, 10, 5)
    
    year_filter = st.sidebar.text_input("Filter by year (optional)")
    year = int(year_filter) if year_filter and year_filter.isdigit() else None
    
    financial_only = st.sidebar.checkbox("Filter for financial info only")
    
    # Conversation controls
    st.sidebar.header("Conversation")
    if st.sidebar.button("Clear Conversation"):
        app = get_rag_application()
        app.clear_conversation()
        st.session_state.messages = []
        st.sidebar.success("Conversation cleared!")
    
    # About section
    st.sidebar.header("About")
    st.sidebar.info(
        "This chatbot uses RAG (Retrieval-Augmented Generation) to answer questions "
        "based on your company documents. It retrieves relevant information and generates "
        "accurate answers using conversational AI."
    )
    
    # Main chat interface
    app = get_rag_application()
    
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Load existing conversation from memory if empty
    if not st.session_state.messages:
        for message in app.get_conversation_history():
            st.session_state.messages.append(message)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat
        st.chat_message("user").markdown(prompt)
        
        # Add to session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.markdown("Thinking...")
            
            # Get the response from RAG application
            response = app.chat(
                query_text=prompt,
                k=k,
                year=year,
                financial_only=financial_only
            )
            
            # Replace placeholder with the response
            response_placeholder.markdown(response)
        
        # Add assistant response to session state
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main() 