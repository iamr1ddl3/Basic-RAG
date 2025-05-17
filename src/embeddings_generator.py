import os
import logging
from typing import List, Dict, Any
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

class EmbeddingsGenerator:
    """Class to generate embeddings for document chunks."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize the embeddings generator.
        
        Args:
            model_name: OpenAI model name for embeddings
        """
        # Use environment variable or default model
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        
        logging.info(f"Initializing OpenAI embeddings model: {self.model_name}")
        
        try:
            # Initialize the embeddings model
            self.embedding_model = OpenAIEmbeddings(
                model=self.model_name,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            logging.info("OpenAI embeddings model initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing embeddings model: {str(e)}")
            raise
    
    def generate_embeddings(self, document_chunks: List[Dict[str, Any]], batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Generate embeddings for document chunks.
        
        Args:
            document_chunks: List of document chunks
            batch_size: Number of chunks to process at once
            
        Returns:
            List of document chunks with embeddings
        """
        if not document_chunks:
            logging.warning("No document chunks provided for embedding generation")
            return []
        
        embedded_chunks = []
        total_chunks = len(document_chunks)
        
        logging.info(f"Generating embeddings for {total_chunks} document chunks")
        
        # Process in batches to avoid memory issues and rate limits
        for i in tqdm(range(0, total_chunks, batch_size), desc="Generating embeddings"):
            batch = document_chunks[i:min(i + batch_size, total_chunks)]
            
            try:
                # Extract text from chunks
                texts = [chunk.page_content for chunk in batch]
                
                # Generate embeddings
                embeddings = self.embedding_model.embed_documents(texts)
                
                # Add embeddings to chunks
                # (Use metadata instead of direct attribute assignment)
                for j, chunk in enumerate(batch):
                    # Create a new dictionary with embedding
                    chunk_with_embedding = {
                        "document": chunk,
                        "embedding": embeddings[j]
                    }
                    embedded_chunks.append(chunk_with_embedding)
                    
            except Exception as e:
                logging.error(f"Error generating embeddings for batch {i//batch_size}: {str(e)}")
        
        logging.info(f"Successfully generated embeddings for {len(embedded_chunks)} chunks")
        return embedded_chunks


if __name__ == "__main__":
    # Example usage
    from document_loader import DocumentProcessor
    
    processor = DocumentProcessor()
    chunks = processor.load_and_split_pdfs("../data")
    
    embedder = EmbeddingsGenerator()
    embedded_chunks = embedder.generate_embeddings(chunks)
    
    print(f"Generated embeddings for {len(embedded_chunks)} chunks") 