import os
from typing import List, Dict, Any
import logging
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentProcessor:
    """Class to handle document loading and preprocessing."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
    
    def load_and_split_pdfs(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Load PDFs from a directory and split them into chunks.
        
        Args:
            directory_path: Path to directory containing PDF files
            
        Returns:
            List of document chunks with metadata
        """
        if not os.path.exists(directory_path):
            logging.error(f"Directory not found: {directory_path}")
            return []
        
        pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            logging.warning(f"No PDF files found in {directory_path}")
            return []
        
        logging.info(f"Found {len(pdf_files)} PDF files in {directory_path}")
        
        all_chunks = []
        
        # Process each PDF file
        for pdf_file in tqdm(pdf_files, desc="Processing PDF files"):
            try:
                file_path = os.path.join(directory_path, pdf_file)
                # Load the PDF
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                
                # Add filename to metadata
                for doc in documents:
                    doc.metadata["source"] = pdf_file
                
                # Split documents into chunks
                chunks = self.text_splitter.split_documents(documents)
                
                # Add document chunks to the collection
                all_chunks.extend(chunks)
                
                logging.info(f"Processed {pdf_file}: {len(chunks)} chunks created")
                
            except Exception as e:
                logging.error(f"Error processing {pdf_file}: {str(e)}")
        
        logging.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks

    def extract_annual_report_metadata(self, document_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract and add specific metadata related to annual reports for each chunk.
        This enhances searchability for financial information.
        
        Args:
            document_chunks: List of document chunks with basic metadata
            
        Returns:
            Enhanced document chunks with financial metadata
        """
        # Keywords that might indicate financial sections
        financial_keywords = [
            "financial statement", "balance sheet", "income statement", 
            "cash flow", "revenue", "profit", "loss", "assets", "liabilities",
            "shareholder", "dividend", "fiscal year", "quarterly report",
            "annual report", "financial performance", "financial results"
        ]
        
        enhanced_chunks = []
        
        for chunk in document_chunks:
            # Check if chunk contains financial information
            contains_financial_info = any(keyword in chunk.page_content.lower() for keyword in financial_keywords)
            
            # Add metadata
            if "metadata" not in chunk:
                chunk.metadata = {}
                
            chunk.metadata["contains_financial_info"] = contains_financial_info
            
            # Try to identify the year or time period
            # This is a simple heuristic and can be improved
            years = []
            for year in range(2000, 2030):
                if str(year) in chunk.page_content:
                    years.append(year)
            
            if years:
                chunk.metadata["years_mentioned"] = years
                
            enhanced_chunks.append(chunk)
            
        return enhanced_chunks


if __name__ == "__main__":
    # Example usage
    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
    chunks = processor.load_and_split_pdfs("../data")
    enhanced_chunks = processor.extract_annual_report_metadata(chunks)
    print(f"Processed {len(enhanced_chunks)} chunks with enhanced metadata") 