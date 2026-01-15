# Personal Memory Assistant

A local AI that learns from your WhatsApp chat history to remember everything about your relationships, decisions, and past conversations. Ask it anything about your personal life - it retrieves relevant memories and answers intelligently.

**100% private. Runs locally. Your data never leaves your machine.**

## What It Does

- **Memory retrieval**: "What did I decide about that apartment?" → Finds relevant past conversations
- **Person insights**: `!about Anna` → Summarizes everything you've discussed with someone
- **Chat search**: `!search [topic]` → Find specific conversations from months ago
- **Decision support**: Analyzes your past concerns to help with current decisions

## Prerequisites

- [Ollama](https://ollama.ai) installed and running
- Python 3.10+
- Poetry for dependency management
- Your WhatsApp chat export as `_chat.txt` in the project folder

## Setup

1. **Export your WhatsApp chat:**
   - Open WhatsApp → Select chat → More options → Export chat
   - Save as `_chat.txt` in this project folder

2. **Install dependencies:**
   ```bash
   poetry install
   poetry shell
   ```

3. **Start Ollama** (in another terminal):
   ```bash
   ollama serve
   ```

4. **Pull a model** (in another terminal):
   ```bash
   ollama pull mistral
   ```

5. **Run the app:**
   ```bash
   python main.py
   ```

## How to Use

### Commands

```
!recall [topic]     - Find all messages about a topic
!about [person]     - Get summary of what you know about someone
!search [query]     - Search your chat history
!stats              - View chat statistics
```

### Regular Chat

Just ask questions naturally - the system finds relevant context from your chat history when helpful.

## Using a Powerful Model (Optional)

By default, this uses Mistral 7B locally. For better results, use Claude or GPT-4 by swapping the LLM in `src/chat.py`:

**Example with Claude:**
```python
from langchain_anthropic import ChatAnthropic

self.llm = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)
```

Then add your API key to `.env` and install: `poetry add langchain-anthropic`

## How It Works

1. Parses your WhatsApp chat export
2. Creates vector embeddings (locally)
3. Stores in local ChromaDB database
4. When you ask a question, retrieves relevant messages
5. LLM answers with context from your history

## Project Structure

- `_chat.txt` - Your WhatsApp chat (excluded from git)
- `chroma_db/` - Local vector database
- `src/data_loader.py` - Parse WhatsApp format
- `src/rag_pipeline.py` - Vector search & retrieval
- `src/chat.py` - Chat interface

## Privacy

Your chat data:
- ✅ Never leaves your machine
- ✅ Stored locally only
- ✅ Works completely offline (except model downloads)
- ✅ You own everything

## Support This Project

If you find this useful:
- 🎵 Into Berlin style techno? [Check out my record](https://decadencerecordings.bandcamp.com/album/d-r-01)
- ☕ Otherwise, [buy me a coffee](https://buymeacoffee.com/shinobe)

## License

MIT

