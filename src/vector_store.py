import os
import logging
import uuid
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from langchain_qdrant import Qdrant
from langchain_openai import OpenAIEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

class VectorStore:
    """Class to manage vector storage in Qdrant."""
    
    def __init__(self, collection_name: str = None, url: str = None, api_key: str = None):
        """
        Initialize connection to Qdrant.
        
        Args:
            collection_name: Name of the Qdrant collection
            url: URL of the Qdrant server
            api_key: API key for Qdrant
        """
        # Use environment variables or defaults
        self.collection_name = collection_name or os.getenv("COLLECTION_NAME", "company_reports")
        self.url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY", None)
        
        logging.info(f"Connecting to Qdrant at {self.url}, collection: {self.collection_name}")
        
        try:
            # Initialize Qdrant client
            self.client = QdrantClient(url=self.url, api_key=self.api_key if self.api_key else None)
            logging.info("Connected to Qdrant successfully")
        except Exception as e:
            logging.error(f"Error connecting to Qdrant: {str(e)}")
            raise
        
        # Initialize embedding model for reuse
        self.embedding_model = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"),
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Try to create collection if it doesn't exist
        self._create_collection_if_not_exists()
    
    def _create_collection_if_not_exists(self, vector_size: int = 1536):
        """
        Create a collection in Qdrant if it doesn't exist.
        
        Args:
            vector_size: Dimension of the embedding vectors (1536 for OpenAI embeddings)
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                logging.info(f"Creating collection {self.collection_name}")
                
                # Create collection with proper configuration
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                
                # Define payload indexes for better search performance
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="metadata.source",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="metadata.contains_financial_info",
                    field_schema=models.PayloadSchemaType.BOOL
                )
                
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="metadata.years_mentioned",
                    field_schema=models.PayloadSchemaType.INTEGER
                )
                
                logging.info(f"Collection {self.collection_name} created successfully")
            else:
                logging.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logging.error(f"Error creating collection: {str(e)}")
            raise
    
    def store_embeddings(self, embedded_chunks: List[Dict[str, Any]]) -> bool:
        """
        Store document embeddings in Qdrant.
        
        Args:
            embedded_chunks: List of document chunks with embeddings
            
        Returns:
            Boolean indicating success
        """
        if not embedded_chunks:
            logging.warning("No embedded chunks provided for storage")
            return False
        
        try:
            # Using the direct Qdrant client API
            logging.info(f"Storing {len(embedded_chunks)} chunks in Qdrant collection {self.collection_name}")
            
            # Prepare points for Qdrant
            points = []
            
            for i, chunk_data in enumerate(embedded_chunks):
                doc = chunk_data["document"]
                embedding = chunk_data["embedding"]
                
                # Create point
                point = models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "text": doc.page_content,
                        "metadata": doc.metadata
                    }
                )
                
                points.append(point)
            
            # Add points in batch
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logging.info(f"Successfully stored {len(embedded_chunks)} embedded chunks in Qdrant")
            return True
            
        except Exception as e:
            logging.error(f"Error storing embeddings in Qdrant: {str(e)}")
            return False
    
    def search(self, query_embedding: List[float], limit: int = 5, filter_params: Optional[Dict] = None) -> List[Dict]:
        """
        Search for similar documents in Qdrant.
        
        Args:
            query_embedding: Embedding vector of the query
            limit: Number of results to return
            filter_params: Optional filter parameters
            
        Returns:
            List of similar documents with scores
        """
        try:
            # Prepare filter if provided
            if filter_params:
                filter_condition = models.Filter(
                    must=[
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        ) for key, value in filter_params.items()
                    ]
                )
            else:
                filter_condition = None
            
            # Search the collection
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                query_filter=filter_condition,
                with_payload=True,
            )
            
            # Format results
            results = []
            for res in search_results:
                result = {
                    "id": res.id,
                    "score": res.score,
                    "payload": res.payload,
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logging.error(f"Error searching in Qdrant: {str(e)}")
            return []


if __name__ == "__main__":
    # Example usage
    from document_loader import DocumentProcessor
    from embeddings_generator import EmbeddingsGenerator
    
    # Load and process documents
    processor = DocumentProcessor()
    chunks = processor.load_and_split_pdfs("../data")
    enhanced_chunks = processor.extract_annual_report_metadata(chunks)
    
    # Generate embeddings
    embedder = EmbeddingsGenerator()
    embedded_chunks = embedder.generate_embeddings(enhanced_chunks)
    
    # Store in Qdrant
    vector_store = VectorStore()
    success = vector_store.store_embeddings(embedded_chunks)
    
    print(f"Storage successful: {success}") 