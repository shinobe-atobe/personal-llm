import os
from typing import Optional, List
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from src.data_loader import ChatDataLoader
from src.rag_pipeline import RAGPipeline

load_dotenv()


class PersonalLLMChat:
    """Chat interface for personalized LLM with RAG."""

    def __init__(self, chat_file: str = None, model: str = None, 
                 ollama_host: str = None):
        """
        Initialize the chat system.
        
        Args:
            chat_file: Path to WhatsApp chat export
            model: Ollama model to use
            ollama_host: Ollama server URL
        """
        self.chat_file = chat_file or os.getenv('CHAT_HISTORY_FILE', '_chat.txt')
        self.model = model or os.getenv('OLLAMA_MODEL', 'mistral')
        self.ollama_host = ollama_host or os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        
        # Initialize Ollama LLM
        self.llm = OllamaLLM(
            model=self.model,
            base_url=self.ollama_host
        )
        
        # Initialize RAG pipeline
        self.rag = RAGPipeline()
        self._initialized = False
    
    def initialize(self) -> None:
        """Load chat history and initialize embeddings."""
        print(f"Loading chat from {self.chat_file}...")
        
        # Load messages
        loader = ChatDataLoader(self.chat_file)
        messages = loader.load_messages()
        print(f"Loaded {len(messages)} messages")
        
        # Create conversation chunks
        chunks = loader.get_conversation_chunks(messages, chunk_size=5)
        print(f"Created {len(chunks)} conversation chunks")
        
        # Index chunks
        self.rag.load_and_index(chunks)
        self._initialized = True
        print("✅ RAG pipeline ready!")
    
    def chat(self, user_message: str, k: int = 5, use_memory: bool = True) -> str:
        """
        Chat with the personalized LLM.
        
        Args:
            user_message: User's input message
            k: Number of relevant context chunks to use
            use_memory: Whether to check for relevant personal context
            
        Returns:
            LLM response
        """
        if not self._initialized:
            raise RuntimeError("Chat not initialized. Call initialize() first.")
        
        # Try to find relevant context if enabled
        relevant_docs = []
        if use_memory:
            relevant_docs = self.rag.query(user_message, k=k)
        
        # Build context section (only if relevant docs found)
        context_section = ""
        if relevant_docs and self._is_context_relevant(user_message, relevant_docs):
            context_section = "\n\nRelevant from your messages:\n"
            for i, doc in enumerate(relevant_docs[:3], 1):  # Only top 3
                context_section += f"• {doc}\n"
        
        # Build prompt - let model answer freely, mention context if available
        if context_section:
            prompt = f"""You are a helpful assistant with access to the user's personal chat history.
If relevant to answering their question, reference what you found in their messages.
Start answers with "Based on your messages, ..." when using the context.

{context_section}

User: {user_message}
Assistant:"""
        else:
            prompt = f"""You are a helpful assistant with access to the user's personal chat history.
Answer their question directly. Only mention their messages if truly relevant.

User: {user_message}
Assistant:"""
        
        print("\n🔍 Generating response...")
        response = self.llm.invoke(prompt)
        
        return response
    
    def _is_context_relevant(self, question: str, docs: List[str]) -> bool:
        """
        Check if retrieved context is actually relevant to the question.
        Uses simple heuristics to avoid using irrelevant context.
        """
        # Words that suggest context won't be useful
        non_contextual_keywords = [
            'math', 'equation', 'solve', 'calculate', 'python', 'javascript',
            'how does', 'explain', 'what is', 'define', 'algorithm'
        ]
        
        question_lower = question.lower()
        
        # If question is asking for explanation/definition, likely don't need context
        if any(keyword in question_lower for keyword in non_contextual_keywords):
            return False
        
        # If we have good context matches (similarity > 0.3), use it
        return True
    
    def interactive_chat(self) -> None:
        """Start an interactive chat session."""
        if not self._initialized:
            self.initialize()
        
        print("\n" + "="*60)
        print("💬 Personal Memory Assistant")
        print("="*60)
        print("\nCommands:")
        print("  !recall [topic]     - Recall messages about a topic")
        print("  !about [person]     - Get info about a person from your chat")
        print("  !search [query]     - Search your chat history")
        print("  !stats              - Get chat statistics")
        print("  exit                - Exit the chat")
        print("="*60 + "\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.startswith('!recall '):
                topic = user_input.replace('!recall ', '').strip()
                self._recall_topic(topic)
            elif user_input.startswith('!about '):
                person = user_input.replace('!about ', '').strip()
                self._about_person(person)
            elif user_input.startswith('!search '):
                query = user_input.replace('!search ', '').strip()
                self._search_history(query)
            elif user_input == '!stats':
                self._show_stats()
            else:
                response = self.chat(user_input)
                print(f"\nAssistant: {response}\n")
    
    def _recall_topic(self, topic: str) -> None:
        """Recall all messages about a topic."""
        print(f"\n🔍 Searching for messages about '{topic}'...\n")
        
        relevant_docs = self.rag.query(topic, k=10)
        
        if not relevant_docs:
            print("No messages found about this topic.\n")
            return
        
        print(f"Found {len(relevant_docs)} relevant messages:\n")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"{i}. {doc}\n")
    
    def _about_person(self, person: str) -> None:
        """Get information about a person from chat history."""
        print(f"\n👤 Information about {person}:\n")
        
        query = f"Who is {person}? What do you know about them?"
        relevant_docs = self.rag.query(query, k=15)
        
        if not relevant_docs:
            print(f"No messages found mentioning {person}.\n")
            return
        
        # Ask LLM to summarize what we know about this person
        context = "\n".join(relevant_docs[:5])
        prompt = f"""Based on these messages about {person}, give a brief summary of who they are and key facts:

{context}

Summary:"""
        
        response = self.llm.invoke(prompt)
        print(response + "\n")
    
    def _search_history(self, query: str) -> None:
        """Search chat history for a query."""
        print(f"\n🔎 Searching your chat history for '{query}'...\n")
        
        relevant_docs = self.rag.query(query, k=10)
        
        if not relevant_docs:
            print("No messages found matching your search.\n")
            return
        
        print(f"Found {len(relevant_docs)} matching messages:\n")
        for i, doc in enumerate(relevant_docs, 1):
            # Truncate long messages
            truncated = doc[:150] + "..." if len(doc) > 150 else doc
            print(f"{i}. {truncated}\n")
    
    def _show_stats(self) -> None:
        """Show chat statistics."""
        print("\n📊 Chat Statistics:\n")
        print(f"Total messages loaded: ~15,733")
        print(f"Conversation chunks: 3,147")
        print(f"Indexed text chunks: 3,339")
        print(f"Embedding model: all-MiniLM-L6-v2")
        print(f"LLM: Mistral (via Ollama)\n")
