import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from langchain_community.vectorstores import Qdrant

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
        self.collection_name = collection_name or os.getenv("COLLECTION_NAME", "technical_manuals")
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