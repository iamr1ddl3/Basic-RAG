import os
import logging
import argparse
from dotenv import load_dotenv

from rag_app import RAGApplication

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

def run_interactive_demo(data_dir: str = None, skip_ingestion: bool = False):
    """
    Run an interactive demo of the RAG application.
    
    Args:
        data_dir: Directory containing PDF files
        skip_ingestion: Whether to skip the document ingestion step
    """
    print("\n" + "=" * 80)
    print(" " * 30 + "RAG SYSTEM DEMO")
    print("=" * 80 + "\n")
    
    # Initialize RAG application
    app = RAGApplication()
    
    # Ingest documents if requested
    if not skip_ingestion and data_dir:
        print(f"\nIngesting documents from directory: {data_dir}\n")
        success = app.ingest_documents(directory_path=data_dir)
        
        if success:
            print("\n✅ Document ingestion successful!\n")
        else:
            print("\n❌ Document ingestion failed!\n")
            return
    elif not skip_ingestion and not data_dir:
        print("\n❌ No data directory provided for ingestion.\n")
        return
    
    # Main interactive loop
    print("\nRAG System is ready! Enter 'exit' at any time to quit.\n")
    print("Available commands:")
    print("  query [text]       - Query the system")
    print("  financial [year]   - Generate financial summary for a specific year")
    print("  help               - Show available commands\n")
    
    while True:
        try:
            # Get user input
            user_input = input("\n> ").strip()
            
            # Check for exit command
            if user_input.lower() in ("exit", "quit"):
                print("\nExiting RAG demo. Goodbye!\n")
                break
                
            # Parse input
            if user_input.lower() == "help":
                print("\nAvailable commands:")
                print("  query [text]       - Query the system")
                print("  financial [year]   - Generate financial summary for a specific year")
                print("  exit/quit          - Exit the demo")
            
            elif user_input.lower().startswith("query "):
                # Extract query text
                query_text = user_input[6:].strip()
                
                if not query_text:
                    print("\n❌ Please provide a query text.")
                    continue
                
                # Process query
                print("\nProcessing query...\n")
                response = app.query(query_text=query_text)
                
                # Display response
                print("\nResponse:")
                print(response)
                
            elif user_input.lower().startswith("financial "):
                # Extract year
                try:
                    year = int(user_input[10:].strip())
                except ValueError:
                    print("\n❌ Please provide a valid year (e.g., financial 2022).")
                    continue
                
                # Generate summary
                print(f"\nGenerating financial summary for {year}...\n")
                summary = app.financial_summary(year=year)
                
                # Display summary
                print("\nFinancial Summary:")
                print(summary)
                
            else:
                print("\n❌ Unknown command. Type 'help' to see available commands.")
                
        except KeyboardInterrupt:
            print("\n\nExiting RAG demo. Goodbye!\n")
            break
            
        except Exception as e:
            logging.error(f"Error in interactive demo: {str(e)}")
            print(f"\n❌ An error occurred: {str(e)}")


def main():
    """Main function to run the interactive demo."""
    parser = argparse.ArgumentParser(description="Interactive RAG System Demo")
    parser.add_argument("--data-dir", help="Directory containing PDF files to ingest")
    parser.add_argument("--skip-ingestion", action="store_true", help="Skip document ingestion")
    
    args = parser.parse_args()
    
    run_interactive_demo(
        data_dir=args.data_dir,
        skip_ingestion=args.skip_ingestion
    )


if __name__ == "__main__":
    main() 