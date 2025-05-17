import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from retriever import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

class Generator:
    """Class to generate responses using retrieved context."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the generator.
        
        Args:
            model_name: OpenAI model to use
        """
        self.model_name = model_name
        
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logging.warning("OpenAI API key not found in environment variables")
            
        try:
            # Initialize LLM
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=0.2,
                api_key=api_key
            )
            logging.info(f"Generator initialized with model: {self.model_name}")
        except Exception as e:
            logging.error(f"Error initializing generator: {str(e)}")
            raise
        
        # Create prompt templates
        self._create_prompt_templates()
    
    def _create_prompt_templates(self):
        """Create prompt templates for different generation tasks."""
        
        # Standard RAG prompt
        self.rag_prompt = ChatPromptTemplate.from_template("""
        You are an AI assistant specialized in providing information about technical manuals and company annual reports.
        Use the following retrieved context to answer the question. If you don't know the answer or can't find it in the context, 
        say that you don't know and avoid making up information.
        
        Context:
        {context}
        
        Question: {question}
        
        When answering:
        1. Provide specific information from the documents when available
        2. Cite the source documents where the information came from
        3. If financial figures are mentioned, be precise with the numbers
        
        Your answer:
        """)
        
        # Conversational RAG prompt with history
        self.conversational_rag_prompt = ChatPromptTemplate.from_template("""
        You are an AI assistant specialized in providing information about technical manuals and company annual reports.
        Use the following retrieved context to answer the latest question. If you don't know the answer or can't find it in the context, 
        say that you don't know and avoid making up information.
        
        Here is the conversation history:
        {conversation_history}
        
        Retrieved context:
        {context}
        
        Latest question: {question}
        
        When answering:
        1. Provide specific information from the documents when available
        2. Cite the source documents where the information came from
        3. If financial figures are mentioned, be precise with the numbers
        4. Be conversational and friendly, but focus on providing accurate information
        5. Only answer the latest question, don't repeat previous answers unless asked to
        
        Your answer:
        """)
        
        # Prompt for summarizing financial information
        self.financial_summary_prompt = ChatPromptTemplate.from_template("""
        You are an AI financial analyst specialized in extracting and summarizing financial information from company annual reports.
        
        Based on the following retrieved context, create a concise summary of the financial performance.
        
        Context:
        {context}
        
        When summarizing:
        1. Focus on key financial metrics (revenue, profit, growth, etc.)
        2. Mention specific time periods and comparisons between periods when available
        3. Highlight any significant changes or trends
        4. Organize the information in a clear, structured way
        5. Cite the source documents for key information
        
        Financial Summary:
        """)
    
    def _format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into context string.
        
        Args:
            documents: List of retrieved Document objects
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            # Format each document with source information
            context_part = f"[Document {i} from {doc.source}]\n{doc.content}\n"
            context_parts.append(context_part)
            
        return "\n".join(context_parts)
    
    def generate_response(self, query: str, documents: List[Document]) -> str:
        """
        Generate a response to a query using retrieved documents.
        
        Args:
            query: User query
            documents: List of retrieved Document objects
            
        Returns:
            Generated response
        """
        if not documents:
            return "I don't have enough information to answer that question."
        
        try:
            # Format context from documents
            context = self._format_context(documents)
            
            # Create and run chain
            chain = (
                {"context": RunnableLambda(lambda x: x["context"]), 
                 "question": RunnableLambda(lambda x: x["question"])}
                | self.rag_prompt
                | self.llm
                | StrOutputParser()
            )
            
            # Run the chain
            response = chain.invoke({"context": context, "question": query})
            
            return response
            
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return f"An error occurred while generating the response: {str(e)}"
    
    def generate_conversational_response(
        self, 
        query: str, 
        documents: List[Document],
        conversation_history: str
    ) -> str:
        """
        Generate a conversational response using retrieved documents and conversation history.
        
        Args:
            query: User's latest query
            documents: List of retrieved Document objects
            conversation_history: String containing conversation history
            
        Returns:
            Generated response
        """
        if not documents:
            return "I don't have enough information to answer that question."
        
        try:
            # Format context from documents
            context = self._format_context(documents)
            
            # Create and run chain
            chain = (
                {
                    "context": RunnableLambda(lambda x: x["context"]),
                    "question": RunnableLambda(lambda x: x["question"]),
                    "conversation_history": RunnableLambda(lambda x: x["conversation_history"])
                }
                | self.conversational_rag_prompt
                | self.llm
                | StrOutputParser()
            )
            
            # Run the chain
            response = chain.invoke({
                "context": context, 
                "question": query,
                "conversation_history": conversation_history
            })
            
            return response
            
        except Exception as e:
            logging.error(f"Error generating conversational response: {str(e)}")
            return f"An error occurred while generating the response: {str(e)}"
    
    def generate_financial_summary(self, documents: List[Document]) -> str:
        """
        Generate a financial summary from retrieved documents.
        
        Args:
            documents: List of retrieved Document objects
            
        Returns:
            Generated financial summary
        """
        if not documents:
            return "No financial information is available to summarize."
        
        try:
            # Format context from documents
            context = self._format_context(documents)
            
            # Create and run chain
            chain = (
                {"context": RunnableLambda(lambda x: x)}
                | self.financial_summary_prompt
                | self.llm
                | StrOutputParser()
            )
            
            # Run the chain
            response = chain.invoke(context)
            
            return response
            
        except Exception as e:
            logging.error(f"Error generating financial summary: {str(e)}")
            return f"An error occurred while generating the financial summary: {str(e)}"


if __name__ == "__main__":
    # Example usage
    from retriever import Retriever, Document
    
    # Create dummy documents for testing
    dummy_docs = [
        Document(
            content="In fiscal year 2022, our company reported revenue of $10.5 million, up 15% from the previous year.",
            source="annual_report_2022.pdf",
            score=0.95,
            metadata={"contains_financial_info": True, "years_mentioned": [2022, 2021]}
        ),
        Document(
            content="Operating expenses increased to $6.2 million, primarily due to expansion into new markets.",
            source="annual_report_2022.pdf",
            score=0.92,
            metadata={"contains_financial_info": True, "years_mentioned": [2022]}
        )
    ]
    
    # Initialize generator
    generator = Generator()
    
    # Generate response to a query
    response = generator.generate_response(
        query="How did the company perform financially in 2022?",
        documents=dummy_docs
    )
    
    print("Generated Response:")
    print(response)
    
    # Generate financial summary
    summary = generator.generate_financial_summary(dummy_docs)
    
    print("\nFinancial Summary:")
    print(summary) 