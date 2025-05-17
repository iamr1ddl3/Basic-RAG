import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

class Document(BaseModel):
    """Model for document records returned by retriever."""
    content: str = Field(..., description="The text content of the document chunk")
    source: str = Field(..., description="Source file of the document")
    score: float = Field(..., description="Similarity score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class Retriever:
    """Class to retrieve relevant documents using vector similarity search."""
    
    def __init__(
        self, 
        collection_name: str = None, 
        url: str = None, 
        api_key: str = None,
        embedding_model: str = None
    ):
        """
        Initialize the retriever with connection to Qdrant and embedding model.
        
        Args:
            collection_name: Name of the Qdrant collection
            url: URL of the Qdrant server
            api_key: API key for Qdrant
            embedding_model: OpenAI model name for embeddings
        """
        # Use environment variables or defaults
        self.collection_name = collection_name or os.getenv("COLLECTION_NAME", "company_reports")
        self.url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY", None)
        self.embedding_model_name = embedding_model or os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        
        logging.info(f"Initializing retriever for collection: {self.collection_name}")
        
        try:
            # Initialize embedding model
            self.embedding_model = OpenAIEmbeddings(
                model=self.embedding_model_name,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            
            # Initialize Qdrant client
            self.client = QdrantClient(url=self.url, api_key=self.api_key if self.api_key else None)
            
            logging.info("Retriever initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing retriever: {str(e)}")
            raise
    
    def retrieve(
        self, 
        query: str, 
        k: int = 5, 
        filter_params: Optional[Dict] = None,
        year: Optional[int] = None,
        financial_only: bool = False
    ) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query string
            k: Number of documents to retrieve
            filter_params: Custom filter parameters
            year: Filter by specific year
            financial_only: Filter for financial information only
            
        Returns:
            List of relevant Document objects
        """
        logging.info(f"Retrieving documents for query: {query}")
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.embed_query(query)
            
            # Build filter based on parameters
            filter_condition = None
            
            if filter_params or year or financial_only:
                must_conditions = []
                
                if filter_params:
                    for key, value in filter_params.items():
                        condition = models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                        must_conditions.append(condition)
                
                if year:
                    condition = models.FieldCondition(
                        key="metadata.years_mentioned",
                        match=models.MatchValue(value=year)
                    )
                    must_conditions.append(condition)
                    
                if financial_only:
                    condition = models.FieldCondition(
                        key="metadata.contains_financial_info",
                        match=models.MatchValue(value=True)
                    )
                    must_conditions.append(condition)
                
                filter_condition = models.Filter(must=must_conditions)
            
            # Search the collection
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k,
                query_filter=filter_condition,
                with_payload=True,
            )
            
            # Format results
            documents = []
            for point in search_results:
                document = Document(
                    content=point.payload.get("text", ""),
                    source=point.payload.get("metadata", {}).get("source", "Unknown"),
                    score=point.score,
                    metadata=point.payload.get("metadata", {})
                )
                documents.append(document)
            
            logging.info(f"Retrieved {len(documents)} relevant documents")
            return documents
            
        except Exception as e:
            logging.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def search_by_filters(
        self,
        financial_only: bool = False,
        year: Optional[int] = None,
        source_file: Optional[str] = None,
        limit: int = 10
    ) -> List[Document]:
        """
        Search documents by metadata filters without a query.
        
        Args:
            financial_only: Filter for financial information only
            year: Filter by specific year
            source_file: Filter by source file
            limit: Maximum number of results
            
        Returns:
            List of Document objects
        """
        try:
            # Build filter based on parameters
            must_conditions = []
            
            if financial_only:
                condition = models.FieldCondition(
                    key="metadata.contains_financial_info",
                    match=models.MatchValue(value=True)
                )
                must_conditions.append(condition)
                
            if year:
                condition = models.FieldCondition(
                    key="metadata.years_mentioned",
                    match=models.MatchValue(value=year)
                )
                must_conditions.append(condition)
                
            if source_file:
                condition = models.FieldCondition(
                    key="metadata.source",
                    match=models.MatchValue(value=source_file)
                )
                must_conditions.append(condition)
            
            filter_condition = models.Filter(must=must_conditions) if must_conditions else None
            
            # Use scroll to get documents matching the filter
            raw_docs = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_condition,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )[0]
            
            # Format results
            documents = []
            for point in raw_docs:
                document = Document(
                    content=point.payload.get("text", ""),
                    source=point.payload.get("metadata", {}).get("source", "Unknown"),
                    score=1.0,  # No score in this case
                    metadata=point.payload.get("metadata", {})
                )
                documents.append(document)
                
            logging.info(f"Found {len(documents)} documents matching filters")
            return documents
            
        except Exception as e:
            logging.error(f"Error searching by filters: {str(e)}")
            return []


if __name__ == "__main__":
    # Example usage
    retriever = Retriever()
    
    # Example query retrieval
    results = retriever.retrieve(
        query="Show me financial performance in 2022",
        k=3,
        financial_only=True,
        year=2022
    )
    
    print(f"Retrieved {len(results)} documents")
    for i, doc in enumerate(results, 1):
        print(f"\nResult {i} (Score: {doc.score:.4f}, Source: {doc.source}):")
        print(f"Content: {doc.content[:150]}...") 