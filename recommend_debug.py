import pandas as pd
import numpy as np
import tensorflow_hub as hub
import faiss 
import time

# Load your dataset
df = pd.read_csv("test.csv")

# Extract product display names
product_display_names = df['productDisplayName'].tolist()

# Load pre-trained Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Precompute embeddings for all product names
product_embeddings = embed(product_display_names)

# Convert embeddings to numpy array
product_embeddings = np.array(product_embeddings)

# Convert to float32 (required by FAISS)
product_embeddings = product_embeddings.astype('float32')

# Build FAISS index
index = faiss.IndexFlatIP(product_embeddings.shape[1])
index.add(product_embeddings)

# User query
user_query = "tshirt"

# Encode user query
query_embedding = embed([user_query])[0].numpy()

# Convert to float32 (required by FAISS)
query_embedding = query_embedding.astype('float32')

# Search for similar products
start_time = time.time()
k = 10  # Number of nearest neighbors to retrieve
distances, indices = index.search(np.expand_dims(query_embedding, axis=0), k)
end_time = time.time()

# Print top k most similar products
for i, idx in enumerate(indices[0]):
    print(f"{i+1}. {product_display_names[idx]}")

print(f"Search time: {end_time - start_time} seconds")
