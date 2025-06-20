from sentence_transformers import SentenceTransformer

documents = []
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
document_embeddings = embedding_model.encode(documents)

