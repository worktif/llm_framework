from sentence_transformers import SentenceTransformer

# Initialize agent models with specific SentenceTransformer configurations
agent_models = {
    "A1": SentenceTransformer('all-MiniLM-L6-v2'),  # Compact model for general-purpose embeddings
    "A2": SentenceTransformer('paraphrase-MiniLM-L6-v2'),  # Model optimized for paraphrase detection
    "A3": SentenceTransformer('paraphrase-albert-small-v2'),  # Lightweight model for paraphrasing tasks
    "A4": SentenceTransformer('all-MiniLM-L12-v2'),  # Larger model with improved accuracy for embeddings
    "A5": SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2'),  # Multilingual model for paraphrasing tasks
}
