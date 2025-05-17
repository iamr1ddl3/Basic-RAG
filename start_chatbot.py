import os
import sys
import time
import subprocess
import logging
import platform
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("chatbot_startup.log")
    ]
)

def ensure_venv():
    """Check if running in a virtual environment and restart if not."""
    # Check if we're already running in a virtual environment
    if sys.prefix == sys.base_prefix:
        # We're not in a virtual environment
        venv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv")
        
        if platform.system() == "Windows":
            python_executable = os.path.join(venv_path, "Scripts", "python.exe")
        else:
            python_executable = os.path.join(venv_path, "bin", "python")
            
        if os.path.exists(python_executable):
            logging.info(f"Restarting using Python from virtual environment: {python_executable}")
            try:
                # Restart the script using the Python from the virtual environment
                os.execv(python_executable, [python_executable] + sys.argv)
            except Exception as e:
                logging.error(f"Failed to restart with virtual environment: {str(e)}")
                # Continue with system Python
        else:
            logging.info("No virtual environment found. Using system Python.")
    else:
        logging.info(f"Using Python from virtual environment: {sys.prefix}")

# Ensure we're running in the virtual environment first
ensure_venv()

# Load environment variables
load_dotenv()

def is_docker_running():
    """Check if Docker is running by running a simple Docker command."""
    try:
        result = subprocess.run(
            ["docker", "info"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            check=False,
            timeout=10
        )
        return result.returncode == 0
    except Exception as e:
        logging.error(f"Error checking Docker status: {str(e)}")
        return False

def start_docker():
    """Attempt to start Docker Desktop on Windows or Docker daemon on other platforms."""
    system = platform.system()
    
    if system == "Windows":
        try:
            # Path to Docker Desktop on Windows
            docker_path = "C:\\Program Files\\Docker\\Docker\\Docker Desktop.exe"
            if os.path.exists(docker_path):
                logging.info("Starting Docker Desktop...")
                subprocess.Popen([docker_path], close_fds=True)
                
                # Wait for Docker to start
                for _ in range(30):  # Try for 30 * 2 = 60 seconds
                    time.sleep(2)
                    if is_docker_running():
                        logging.info("Docker Desktop started successfully")
                        return True
                    logging.info("Waiting for Docker to start...")
                
                logging.warning("Docker Desktop started but Docker daemon is not responding")
                return False
            else:
                logging.error(f"Docker Desktop not found at {docker_path}")
                return False
        except Exception as e:
            logging.error(f"Error starting Docker Desktop: {str(e)}")
            return False
    else:
        # For Linux/Mac, try to start Docker daemon
        try:
            logging.info("Attempting to start Docker daemon...")
            subprocess.run(["sudo", "systemctl", "start", "docker"], check=True)
            time.sleep(5)
            return is_docker_running()
        except Exception as e:
            logging.error(f"Error starting Docker daemon: {str(e)}")
            return False

def is_qdrant_container_running():
    """Check if the Qdrant container is running."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=qdrant", "--format", "{{.Names}}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True
        )
        return "qdrant" in result.stdout
    except Exception as e:
        logging.error(f"Error checking Qdrant container status: {str(e)}")
        return False

def start_qdrant():
    """Start the Qdrant container using docker-compose."""
    try:
        logging.info("Starting Qdrant container...")
        subprocess.run(
            ["docker-compose", "up", "-d"],
            check=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        # Verify Qdrant is running
        for _ in range(5):  # Try for 5 seconds
            time.sleep(1)
            if is_qdrant_container_running():
                logging.info("Qdrant container started successfully")
                return True
        
        logging.warning("Qdrant container may not have started properly")
        return False
    except Exception as e:
        logging.error(f"Error starting Qdrant container: {str(e)}")
        return False

def start_streamlit():
    """Start the Streamlit application."""
    try:
        logging.info("Starting Streamlit application...")
        
        # Get the project root (where this script is located)
        project_root = os.path.dirname(os.path.abspath(__file__))
        
        # Define explicit paths
        src_path = os.path.join(project_root, "src")
        app_path = os.path.join(src_path, "chatbot_app.py")
        
        # Get streamlit from venv
        venv_dir = os.path.join(project_root, "venv")
        if platform.system() == "Windows":
            streamlit_exe = os.path.join(venv_dir, "Scripts", "streamlit.exe")
        else:
            streamlit_exe = os.path.join(venv_dir, "bin", "streamlit")
            
        logging.info(f"Running: {streamlit_exe} run {app_path}")
        
        # Check if files exist
        if not os.path.exists(streamlit_exe):
            logging.error(f"Streamlit not found at: {streamlit_exe}")
            # Fall back to command-line streamlit if available
            streamlit_exe = "streamlit"
        
        if not os.path.exists(app_path):
            raise FileNotFoundError(f"Chatbot app not found at: {app_path}")
            
        # Run streamlit
        subprocess.run(
            [streamlit_exe, "run", app_path],
            check=True
        )
        return True
    except Exception as e:
        logging.error(f"Error starting Streamlit application: {str(e)}")
        return False

def main():
    """Main function to run the startup script."""
    logging.info("Starting RAG Chatbot...")
    
    # Check if Docker is running
    if not is_docker_running():
        logging.warning("Docker is not running. Attempting to start Docker...")
        if not start_docker():
            logging.error("Failed to start Docker. Please start Docker manually and try again.")
            print("\nERROR: Docker could not be started. Please start Docker Desktop manually and run this script again.")
            return False
    
    # Check if Qdrant container is running
    if not is_qdrant_container_running():
        logging.info("Qdrant container is not running. Starting it...")
        if not start_qdrant():
            logging.error("Failed to start Qdrant container.")
            print("\nERROR: Could not start the Qdrant container. Check Docker is running correctly.")
            return False
    
    # Start Streamlit application
    return start_streamlit()

if __name__ == "__main__":
    if main():
        logging.info("Chatbot startup completed successfully")
    else:
        logging.error("Chatbot startup failed")
        sys.exit(1) 