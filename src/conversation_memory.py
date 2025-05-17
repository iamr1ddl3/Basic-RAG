import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ConversationMemory:
    """Class to manage conversation history for the chatbot."""
    
    def __init__(self, max_history: int = 10):
        """
        Initialize the conversation memory.
        
        Args:
            max_history: Maximum number of messages to keep in history
        """
        self.messages = []
        self.max_history = max_history
        logging.info(f"Conversation memory initialized with max_history={max_history}")
    
    def add_user_message(self, message: str) -> None:
        """
        Add a user message to the conversation history.
        
        Args:
            message: User's message
        """
        self.messages.append({
            "role": "user",
            "content": message
        })
        
        # Trim history if needed
        self._trim_history()
    
    def add_assistant_message(self, message: str) -> None:
        """
        Add an assistant message to the conversation history.
        
        Args:
            message: Assistant's message
        """
        self.messages.append({
            "role": "assistant",
            "content": message
        })
        
        # Trim history if needed
        self._trim_history()
    
    def get_history(self) -> List[Dict[str, str]]:
        """
        Get the full conversation history.
        
        Returns:
            List of message dictionaries
        """
        return self.messages
    
    def get_context_string(self, num_messages: Optional[int] = None) -> str:
        """
        Get a formatted string of recent conversation history for context.
        
        Args:
            num_messages: Number of recent messages to include (or all if None)
            
        Returns:
            Formatted conversation context
        """
        if num_messages is None or num_messages >= len(self.messages):
            messages_to_include = self.messages
        else:
            messages_to_include = self.messages[-num_messages:]
        
        context_parts = []
        
        for msg in messages_to_include:
            role = "User" if msg["role"] == "user" else "Assistant"
            context_parts.append(f"{role}: {msg['content']}")
        
        return "\n".join(context_parts)
    
    def clear(self) -> None:
        """Clear the conversation history."""
        self.messages = []
        logging.info("Conversation memory cleared")
    
    def _trim_history(self) -> None:
        """Trim history to the maximum length."""
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:] 