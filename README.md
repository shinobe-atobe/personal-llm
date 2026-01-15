# Personal LLM with RAG

A personalized memory assistant that uses Retrieval-Augmented Generation (RAG) on your WhatsApp chat history. Keeps your data local while intelligently retrieving relevant personal context to answer your questions.

## 🎯 Use Cases

- **Memory augmentation**: "What did I decide about that apartment?" - instantly retrieve relevant past conversations
- **Person insights**: `!about Anna` - summarize everything you've discussed with someone
- **Decision support**: Analyze your past concerns and opinions to help with current decisions
- **Chat history search**: `!search [topic]` - find specific conversations from months ago
- **Personal knowledge base**: Extract decisions, plans, and information you've shared

## 🔒 Privacy & Data Security

**Your data never leaves your machine:**
- ✅ Chat file stays local
- ✅ Vector embeddings generated locally
- ✅ All processing happens on your device
- ✅ No cloud storage or external API calls (unless you choose hosted model option)
- ✅ You own all your data - delete anytime

Unlike ChatGPT/Claude APIs, this system:
- Processes sensitive personal messages locally
- Doesn't send raw chat history anywhere
- Works completely offline (except Ollama model download)

## Prerequisites

- [Ollama](https://ollama.ai) installed (for local model) OR Claude/OpenAI API key (for hosted)
- Python 3.10+
- Poetry for dependency management

## Setup

1. **Clone Poetry environment:**
   ```bash
   poetry install
   ```

2. **Activate virtual environment:**
   ```bash
   poetry shell
   ```

3. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration (see options below)
   ```

4. **For Local Model (Ollama):**
   ```bash
   # Terminal 1: Start Ollama
   ollama serve
   
   # Terminal 2: Pull model
   ollama pull mistral
   
   # Terminal 3: Run the app
   python main.py
   ```

5. **For Hosted Model (Claude/OpenAI):**
   - Get API key from [Anthropic](https://console.anthropic.com) or [OpenAI](https://platform.openai.com)
   - Add to `.env`: `ANTHROPIC_API_KEY=sk-ant-...` or `OPENAI_API_KEY=sk-...`
   - See "Using Hosted Models" section below

## Usage

### Interactive Commands

```
!recall [topic]     - Find all messages about a topic
!about [person]     - Get summary of what you know about someone
!search [query]     - Search your chat history
!stats              - View chat statistics
```

### Regular Chat

Just ask questions - the system intelligently uses relevant context only when needed.

## Architecture

```
Your WhatsApp chat (_chat.txt)
    ↓
Local parsing & chunking
    ↓
Vector embeddings (local)
    ↓
ChromaDB (local vector storage)
    ↓
Smart context retrieval
    ↓
LLM (local OR cloud)
    ↓
Response with sources
```

## 🌐 Using Hosted Models

### Option 1: Claude (Recommended)

**Why Claude?** Best at personality analysis and instruction following.

1. Get API key from [console.anthropic.com](https://console.anthropic.com)
2. Update `.env`:
   ```
   ANTHROPIC_API_KEY=sk-ant-your-key-here
   ```
3. Edit `src/chat.py` - replace the Ollama initialization:
   ```python
   from langchain_anthropic import ChatAnthropic
   
   # In __init__:
   self.llm = ChatAnthropic(
       model="claude-3-sonnet-20240229",
       api_key=os.getenv("ANTHROPIC_API_KEY")
   )
   ```
4. Install: `poetry add langchain-anthropic`

**Cost:** ~$0.01 per query (~$3/month for daily use)

### Option 2: OpenAI (GPT-4)

1. Get API key from [platform.openai.com](https://platform.openai.com)
2. Update `.env`:
   ```
   OPENAI_API_KEY=sk-your-key-here
   ```
3. Edit `src/chat.py`:
   ```python
   from langchain_openai import ChatOpenAI
   
   # In __init__:
   self.llm = ChatOpenAI(
       model="gpt-4",
       api_key=os.getenv("OPENAI_API_KEY")
   )
   ```
4. Install: `poetry add langchain-openai`

**Cost:** ~$0.03-0.05 per query (~$10/month for daily use)

**Important:** RAG stays local - only short context chunks are sent to API, never your raw chat file.

## Project Structure

- `_chat.txt` - Your WhatsApp chat history (used for RAG)
- `chroma_db/` - Local vector database (created after first run)
- `.env` - Configuration (your API keys if using hosted model)
- `src/`:
  - `data_loader.py` - Parse WhatsApp exports
  - `rag_pipeline.py` - Vector embeddings & retrieval
  - `chat.py` - Main chat interface

## Technologies

- **Ollama** - Local LLM inference
- **LangChain** - LLM framework & RAG
- **ChromaDB** - Vector database (local)
- **Sentence Transformers** - Embedding generation (local)
- **Anthropic Claude** OR **OpenAI** - Optional hosted models
- **Python-dotenv** - Configuration management

## License

MIT

