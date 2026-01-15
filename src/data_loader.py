import re
from typing import List, Tuple
from pathlib import Path


class ChatDataLoader:
    """Load and parse WhatsApp chat export."""

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)

    def load_messages(self) -> List[Tuple[str, str, str]]:
        """
        Load and parse WhatsApp messages.
        
        Returns:
            List of tuples: (timestamp, sender, message)
        """
        messages = []
        
        if not self.filepath.exists():
            raise FileNotFoundError(f"Chat file not found: {self.filepath}")
        
        with open(self.filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # WhatsApp format: [DD.MM.YY, HH:MM:SS] Sender: Message
        pattern = r'\[([^\]]+)\]\s+([^:]+):\s+(.*)'
        
        current_message = None
        
        for line in lines:
            line = line.rstrip('\n')
            
            # Skip empty lines and system messages
            if not line or 'Messages and calls are end-to-end encrypted' in line:
                continue
            
            match = re.match(pattern, line)
            if match:
                timestamp, sender, message = match.groups()
                # Skip media omitted messages and system notifications
                if self._should_skip_message(message):
                    continue
                messages.append((timestamp, sender, message))
            elif current_message:
                # Continuation of previous message (multi-line)
                messages[-1] = (messages[-1][0], messages[-1][1], 
                               messages[-1][2] + ' ' + line)
        
        return messages
    
    @staticmethod
    def _should_skip_message(message: str) -> bool:
        """Skip system messages and media omitted."""
        skip_patterns = [
            'image omitted',
            'audio omitted',
            'video omitted',
            'document omitted',
            'sticker omitted',
            'Missed voice call',
            'Missed video call',
            'Messages and calls are end-to-end encrypted',
        ]
        return any(pattern in message for pattern in skip_patterns)
    
    def get_conversation_chunks(self, messages: List[Tuple[str, str, str]], 
                               chunk_size: int = 5) -> List[str]:
        """
        Group messages into conversation chunks for better context.
        
        Args:
            messages: List of (timestamp, sender, message) tuples
            chunk_size: Number of messages per chunk
            
        Returns:
            List of formatted conversation chunks
        """
        chunks = []
        
        for i in range(0, len(messages), chunk_size):
            chunk_messages = messages[i:i + chunk_size]
            chunk_text = self._format_chunk(chunk_messages)
            if chunk_text.strip():
                chunks.append(chunk_text)
        
        return chunks
    
    @staticmethod
    def _format_chunk(messages: List[Tuple[str, str, str]]) -> str:
        """Format a chunk of messages as readable text."""
        formatted = []
        for timestamp, sender, message in messages:
            formatted.append(f"{sender}: {message}")
        return '\n'.join(formatted)
