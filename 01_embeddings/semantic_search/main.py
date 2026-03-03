
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load embedding model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

query = "who is fast bowler"

docs = [
    "Virat Kohli is an aggressive batsman known for his passionate celebrations.",
    "Rohit Sharma is a stylish opening batsman with excellent timing.",
    "Jasprit Bumrah is a fast bowler famous for his unique action and yorkers.",
    "Hardik Pandya is an all-rounder who contributes with both bat and ball.",
    "MS Dhoni is a calm captain and a reliable finisher in limited-overs cricket.",
    "Sachin Tendulkar is regarded as one of the greatest batsmen in cricket history.",
    "Ravindra Jadeja is a dependable all-rounder and an exceptional fielder.",
    "KL Rahul is a technically sound batsman who can adapt to different formats.",
    "Shubman Gill is a young and talented top-order batsman.",
    "Mohammed Shami is a skilled fast bowler known for his seam movement."
]

# Generate embeddings
doc_embeddings = embedding.embed_documents(docs)
query_embedding = embedding.embed_query(query)

# Compute similarity
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

# Get best match
best_index = np.argmax(scores)
best_score = scores[best_index]

print("Query:", query)
print("Best Match:", docs[best_index])
print("Score:", best_score)
