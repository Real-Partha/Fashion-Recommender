import os
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Function to encode text using BERT tokenizer
def encode_text(text):
    tokens = tokenizer.tokenize(text)
    tokens = tokens[:tokenizer.model_max_length - 2]  # Limiting to BERT's maximum input length
    input_ids = tokenizer.encode(tokens, add_special_tokens=True)
    return input_ids

# Function to calculate similarity
def calculate_similarity(model, user_query_embedding, product_embeddings):
    similarities = cosine_similarity(user_query_embedding, product_embeddings)
    return similarities

# Check if model exists
model_path = "bert_similarity_model.pth"
if os.path.exists(model_path):
    model = torch.load(model_path)
else:
    # Load your dataset
    df = pd.read_csv("your_dataset.csv")

    # Extract product display names
    product_display_names = df['product_display_name'].tolist()

    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Encode product display names
    encoded_product_display_names = [encode_text(name) for name in product_display_names]

    # Encode all product display names and save their embeddings
    product_embeddings = []
    with torch.no_grad():
        num_samples = len(encoded_product_display_names)
        print("Training model:")
        for i, product_name in enumerate(encoded_product_display_names):
            if (i + 1) % 100 == 0 or i == num_samples - 1:
                print(f"Processed {i+1}/{num_samples} samples")
            product_tensor = torch.tensor(product_name).unsqueeze(0)
            product_embedding = model(product_tensor)[0][:, 0, :].numpy()
            product_embeddings.append(product_embedding)
    product_embeddings = np.vstack(product_embeddings)  # Convert list of embeddings to numpy array

    # Save the model
    torch.save(model, model_path)
    print("Model training completed.")

# Encode user query
user_query = "your user query here"
encoded_user_query = encode_text(user_query)

# Encode user query and calculate similarity
with torch.no_grad():
    user_query_tensor = torch.tensor(encoded_user_query).unsqueeze(0)
    user_query_embedding = model(user_query_tensor)[0][:, 0, :].numpy()  # Take the embedding of [CLS] token

# Calculate similarity
similarities = calculate_similarity(model, user_query_embedding, product_embeddings)

# Rank results
k = 10  # Number of similar products to retrieve
top_k_indices = similarities.argsort()[0][-k:][::-1]
top_k_products = [product_display_names[index] for index in top_k_indices]

# Print top k most similar products
print("Top similar products:")
for i, product_name in enumerate(top_k_products):
    print(f"{i+1}. {product_name}")
