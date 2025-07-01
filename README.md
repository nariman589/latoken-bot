# LATOKEN Candidate Assistant Bot 🧠🤖

**🏆 Prize-winning hackathon project** — built in 48 hours to assist job candidates applying to LATOKEN.  
This Telegram bot uses real-time NLP with OpenAI GPT-4 and semantic search to answer questions and generate quiz-style tests based on LATOKEN’s hiring materials.

## 🚀 What It Does
- 🧠 Scrapes public LATOKEN content using Puppeteer (including from dynamic Coda pages)
- 🔎 Builds a semantic search index with OpenAI Embeddings (`text-embedding-3-small`)
- 💬 Answers user questions using RAG (retrieval-augmented generation) with GPT-4
- ✅ Generates multiple-choice questions from real LATOKEN content
- 💾 Caches raw page content for speed and efficiency
- 🤖 Runs as a Telegram bot for candidate interaction

## 💡 Why It Matters

This project demonstrates:
- Building an end-to-end NLP pipeline from scratch
- Integrating OpenAI for embeddings and chat completion
- Implementing a custom vector search engine with cosine similarity
- Scalable web scraping using headless Chrome
- Practical use of RAG for user-facing applications

## 🧱 Tech Stack

- **Node.js** — backend logic
- **Puppeteer** — web scraping
- **OpenAI API** — GPT-4 + embeddings
- **Telegraf** — Telegram bot framework
- **Custom vector search** — similarity-based retrieval
- **Filesystem caching** — for embeddings & scraped pages

## 📦 Folder Structure
├── addManualData.js
├── data/ # Knowledge base and embeddings
├── cache/ # Cached page content
├── index.js # Main app logic

## ▶️ Usage

1. Clone the repo
2. Create a `.env` with your `OPENAI_API_KEY` and `TELEGRAM_TOKEN`
3. Run the bot:

bash
npm install
node index.js

🧪 Test Mode
Use /test in Telegram to toggle test generation mode. The bot will:

Answer your query

Generate a quiz-style question from the knowledge base

⚠️ Notes
Uses GPT-4 for RAG (you can downgrade to GPT-3.5 if needed)

No database used — everything is stored locally for simplicity

LATOKEN content is gathered from public sources only

## 🏆 Hackathon Achievement

This project was developed for a LATOKEN hackathon and won a **top prize** for:
- Innovation in candidate engagement
- Real-world application of RAG (retrieval-augmented generation)
- End-to-end system delivery (scraper + AI + Telegram bot)
