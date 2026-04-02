# Cosmetic Formulation Engine
AuraFormulate is an end-to-end AI lab for cosmetic product formulation. 
Describe a product in plain English — "cherry lip balm for dry lips" — and the system extracts your intent using zero-shot NLP, 
retrieves the closest matching formulation from a FAISS vector index, explains every ingredient with verified sources (FDA, CIR, WHO), 
checks safety flags, and benchmarks your formula against real competitor products like Carmex, EOS, and CeraVe.
Built with a Flask backend, vanilla JS frontend, and a semantic search pipeline powered by sentence-transformers and faiss-cpu.
