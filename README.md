# YouTube Q\&A Chatbot with RAG

A powerful Streamlit-based chatbot that allows you to ask questions about the content of any YouTube video. It uses **Gemini 2.0 Flash** for transcription and answer generation, and **Pinecone** vector database for RAG (Retrieval-Augmented Generation).

---

## 🚀 Features

* 🎥 Load any YouTube video and extract transcript or audio
* 🧠 Ask questions and get context-aware answers from the video
* 🌐 Multi-language support (25+ languages)
* 🔍 RAG pipeline using Pinecone + Sentence Transformers
* 🗣️ Voice input with real-time recording & transcription
* 🗃️ Auto-save and reuse vectorized content for faster future access
* 🤖 Powered by **Gemini API** for transcription and Q\&A

---

## 📦 Tech Stack

* **Frontend**: Streamlit
* **Backend**: Python
* **AI Models**:

  * Gemini 2.0 Flash (`google.generativeai`)
  * Sentence Transformers (`all-MiniLM-L6-v2`)
  * `tiktoken` for accurate chunking
* **Audio Processing**:

  * `sounddevice`, `wavio`, `speech_recognition`
* **Video Handling**:

  * `yt_dlp`, `youtube_transcript_api`
* **Vector Store**:

  * Pinecone
* **Env Management**:

  * `dotenv`

---

## 🛠️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/youtube-qa-chatbot.git
cd youtube-qa-chatbot
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Create a **\`\`** file** with the following:

```env
GEMINI_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

Then go to `http://localhost:8501` in your browser.

---

## 📂 Directory Structure

```
youtube-qa-chatbot/
├── main.py           # Main streamlit app
├── requirements.txt
└── .env
```

---

## ❓ How It Works

1. **Load a video**

   * Extracts transcript using YouTube API or downloads audio and transcribes it using Gemini.

2. **Chunks and Embeddings**

   * Splits the transcript into chunks using `tiktoken`, generates embeddings using Sentence Transformers.

3. **RAG Pipeline**

   * Stores embeddings in Pinecone
   * For each user query, finds relevant chunks and uses Gemini to answer based on them.

4. **Chat Interface**

   * Text or voice-based input
   * Gemini answers in the language of the question when possible

---

## 🔐 .gitignore Example

```gitignore
.env
__pycache__/
*.pyc
*.wav
temp_audio/
.vscode/
*.log
```

---

## 📢 Credits

* Built with OpenAI GPT & Gemini APIs
* Powered by Pinecone vector database
* Inspired by modern GenAI apps & RAG pipelines

---

## 📜 License

MIT License. Feel free to use and modify this project for educational or commercial use. Attribution appreciated!
