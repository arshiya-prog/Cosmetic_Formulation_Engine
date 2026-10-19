# Cosmetic Formulation Engine
AuraFormulate is an end-to-end AI lab for cosmetic product formulation.
Describe a product in plain English, such as "cherry lip balm for dry lips", and the system extracts your intent using zero-shot NLP,
retrieves the best matching formulations and ingredients using a FAISS vector index with hybrid re-ranking, lets a local LLM generate a grounded formula,
explains every ingredient with verified sources (FDA, CIR, WHO), checks safety flags, and benchmarks your formula against real competitor products like Carmex, EOS, and CeraVe.
Built with a Flask backend, vanilla JS frontend, a semantic search pipeline powered by sentence-transformers and faiss-cpu, and Llama 3.1 8B running locally via Ollama.

## Features
* 🔍 Ingredient-based recommendation engine
* 🧠 Semantic search using Sentence Transformers
* ⚡ Fast similarity search with FAISS, re-ranked by product type, skin concern, texture and flavor
* 🤖 Retrieval-Augmented Generation (RAG) with a local LLM (Llama 3.1 8B via Ollama)
* 🛡️ Safety-aware generation: ingredients unsuitable for the skin concern are kept out
* 🧴 Cosmetic/food formulation support
* 📊 Structured ingredient dataset (expandable)
* 🌐 Frontend + Backend architecture

## Project Structure
```
ingredient-recommendation-system/
│
├── backend.py
├── index.html
├── login.html
├── requirements.txt
│
├── data/
│   ├── archive.csv
│   └── formulations.csv
│
├── modules/
│   ├── description_generator.py
│   ├── explainability.py
│   ├── formulation_model.py
│   ├── nlp_extractor.py
│   ├── rag_generator.py
│   └── safety_checker.py
│
└── README.md
```

## Setup
Install Python dependencies:
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Install Ollama and download the LLM (one-time, about 4.9 GB):
```
brew install ollama
brew services start ollama
ollama pull llama3.1:8b
```
If Ollama is not running, the app still works and falls back to template-based generation.

## Running
To run backend:
```
python backend.py
```
http://127.0.0.1:5000

To run frontend:
```
python -m http.server 9000
```
http://127.0.0.1:9000
