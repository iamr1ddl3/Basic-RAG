import os
import argparse
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from document_loader import DocumentProcessor
from embeddings_generator import EmbeddingsGenerator
from vector_store import VectorStore
from retriever import Retriever, Document
from generator import Generator
from conversation_memory import ConversationMemory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

class RAGApplication:
    """Main application class that integrates all RAG components."""
    
    def __init__(self):
        """Initialize the RAG application."""
        # Get environment variables
        self.chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 200))
        self.collection_name = os.getenv("COLLECTION_NAME", "company_reports")
        
        # Initialize components
        self.document_processor = DocumentProcessor(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        self.embeddings_generator = EmbeddingsGenerator()
        self.vector_store = VectorStore(collection_name=self.collection_name)
        self.retriever = Retriever(collection_name=self.collection_name)
        self.generator = Generator()
        
        # Initialize conversation memory
        self.conversation_memory = ConversationMemory(max_history=20)
        
        logging.info("RAG application initialized successfully")
    
    def ingest_documents(self, directory_path: str, process_metadata: bool = True) -> bool:
        """
        Ingest documents from a directory into the vector store.
        
        Args:
            directory_path: Path to directory containing PDF files
            process_metadata: Whether to extract and add metadata
            
        Returns:
            Boolean indicating success
        """
        try:
            logging.info(f"Starting document ingestion from {directory_path}")
            
            # Load and split documents
            chunks = self.document_processor.load_and_split_pdfs(directory_path)
            
            if not chunks:
                logging.warning("No chunks created from documents")
                return False
            
            # Extract metadata if requested
            if process_metadata:
                chunks = self.document_processor.extract_annual_report_metadata(chunks)
            
            # Generate embeddings
            embedded_chunks = self.embeddings_generator.generate_embeddings(chunks)
            
            if not embedded_chunks:
                logging.warning("No embeddings generated")
                return False
            
            # Store in vector store
            success = self.vector_store.store_embeddings(embedded_chunks)
            
            if success:
                logging.info(f"Successfully ingested {len(embedded_chunks)} document chunks")
            else:
                logging.error("Failed to store embeddings in vector store")
                
            return success
            
        except Exception as e:
            logging.error(f"Error during document ingestion: {str(e)}")
            return False
    
    def query(
        self, 
        query_text: str, 
        k: int = 5, 
        year: Optional[int] = None,
        financial_only: bool = False
    ) -> str:
        """
        Process a query and generate a response.
        
        Args:
            query_text: User query
            k: Number of documents to retrieve
            year: Optional year to filter by
            financial_only: Whether to filter for financial info only
            
        Returns:
            Generated response
        """
        try:
            logging.info(f"Processing query: {query_text}")
            
            # Retrieve relevant documents
            documents = self.retriever.retrieve(
                query=query_text,
                k=k,
                year=year,
                financial_only=financial_only
            )
            
            if not documents:
                return "No relevant documents found to answer your query."
            
            # Generate response
            response = self.generator.generate_response(
                query=query_text,
                documents=documents
            )
            
            return response
            
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            return f"An error occurred while processing your query: {str(e)}"
    
    def chat(
        self, 
        query_text: str, 
        k: int = 5, 
        year: Optional[int] = None,
        financial_only: bool = False,
        conversation_context_size: int = 5
    ) -> str:
        """
        Process a conversational query and generate a response.
        
        Args:
            query_text: User query
            k: Number of documents to retrieve
            year: Optional year to filter by
            financial_only: Whether to filter for financial info only
            conversation_context_size: How many previous messages to include
            
        Returns:
            Generated response
        """
        try:
            logging.info(f"Processing conversational query: {query_text}")
            
            # Add the user query to conversation memory
            self.conversation_memory.add_user_message(query_text)
            
            # Retrieve relevant documents
            documents = self.retriever.retrieve(
                query=query_text,
                k=k,
                year=year,
                financial_only=financial_only
            )
            
            if not documents:
                response = "No relevant documents found to answer your query."
                self.conversation_memory.add_assistant_message(response)
                return response
            
            # Get conversation history string
            conversation_history = self.conversation_memory.get_context_string(conversation_context_size)
            
            # Generate conversational response
            response = self.generator.generate_conversational_response(
                query=query_text,
                documents=documents,
                conversation_history=conversation_history
            )
            
            # Add the response to conversation memory
            self.conversation_memory.add_assistant_message(response)
            
            return response
            
        except Exception as e:
            logging.error(f"Error processing conversational query: {str(e)}")
            error_response = f"An error occurred while processing your query: {str(e)}"
            self.conversation_memory.add_assistant_message(error_response)
            return error_response
    
    def clear_conversation(self) -> None:
        """Clear the conversation history."""
        self.conversation_memory.clear()
        logging.info("Conversation history cleared")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the conversation history.
        
        Returns:
            List of message dictionaries
        """
        return self.conversation_memory.get_history()
    
    def financial_summary(self, year: Optional[int] = None, k: int = 10) -> str:
        """
        Generate a financial summary for a specific year.
        
        Args:
            year: Year to generate summary for
            k: Number of documents to retrieve
            
        Returns:
            Generated financial summary
        """
        try:
            # Get financial documents
            documents = self.retriever.search_by_filters(
                financial_only=True,
                year=year,
                limit=k
            )
            
            if not documents:
                return f"No financial information found{' for ' + str(year) if year else ''}."
            
            # Generate summary
            summary = self.generator.generate_financial_summary(documents)
            
            return summary
            
        except Exception as e:
            logging.error(f"Error generating financial summary: {str(e)}")
            return f"An error occurred while generating the financial summary: {str(e)}"


def main():
    """Main function to run the RAG application."""
    parser = argparse.ArgumentParser(description="RAG Application for PDF Processing")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("--dir", required=True, help="Directory containing PDF files")
    ingest_parser.add_argument("--no-metadata", action="store_true", help="Skip metadata extraction")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the system")
    query_parser.add_argument("--text", required=True, help="Query text")
    query_parser.add_argument("--k", type=int, default=5, help="Number of documents to retrieve")
    query_parser.add_argument("--year", type=int, help="Filter by year")
    query_parser.add_argument("--financial", action="store_true", help="Filter for financial info only")
    
    # Summary command
    summary_parser = subparsers.add_parser("summary", help="Generate financial summary")
    summary_parser.add_argument("--year", type=int, help="Year to summarize")
    summary_parser.add_argument("--k", type=int, default=10, help="Number of documents to retrieve")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize RAG application
    app = RAGApplication()
    
    # Execute command
    if args.command == "ingest":
        success = app.ingest_documents(
            directory_path=args.dir,
            process_metadata=not args.no_metadata
        )
        print(f"Ingestion {'successful' if success else 'failed'}")
        
    elif args.command == "query":
        response = app.query(
            query_text=args.text,
            k=args.k,
            year=args.year,
            financial_only=args.financial
        )
        print("\nResponse:")
        print(response)
        
    elif args.command == "summary":
        summary = app.financial_summary(
            year=args.year,
            k=args.k
        )
        print("\nFinancial Summary:")
        print(summary)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 