import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import random

model = SentenceTransformer("all-MiniLM-L6-v2")


def load_data():
    df = pd.read_csv("data/formulations.csv")

    # Ensure flavor column exists
    if "flavor" not in df.columns:
        df["flavor"] = "neutral"

    # Convert structured data to NL
    df["text"] = df.apply(
        lambda r: f"{r['product_type']} {r['skin_concern']} {r['texture']} {r['color']} {r['flavor']}",
        axis=1
    )
    return df


def build_index(df):
    # Rows are converted to embeddings
    embeddings = model.encode(df["text"].tolist(), convert_to_numpy=True)
    # Normalize embeddings to unit vectors. Cuz dot prod = cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Create FAISS index, IP = Inner Product, Because of normalization → behaves like cosine similarity
    index = faiss.IndexFlatIP(embeddings.shape[1])
    # Store vectors. Now FAISS holds all formulations in vector space
    index.add(embeddings)

    return index


def diversify_ingredients(ingredients_str, flavor):
    ingredients = [i.strip() for i in ingredients_str.split(",")]

    flavor_map = {
        "cherry": "cherry extract",
        "blueberry": "blueberry extract",
        "strawberry": "strawberry extract",
        "mint": "mint extract",
        "vanilla": "vanilla extract"
    }

    if flavor in flavor_map:
        ingredients.append(flavor_map[flavor])

    return ", ".join(list(set(ingredients)))


def retrieve_formulation(attributes: dict):
    df = load_data()
    index = build_index(df)

    # Build query text
    query = f"{attributes.get('product_type','')} {attributes.get('skin_concern','')} {attributes.get('texture','')} {attributes.get('color','')} {attributes.get('flavor','')}"
    
    # Normalize query to unit vector. Now query is a point in the same 384D space
    q_vec = model.encode([query], convert_to_numpy=True)
    q_vec = q_vec / np.linalg.norm(q_vec)

    # Compute inner product (= cosine similarity) via FAISS
    # Top-3 retrieval for variation
    scores, idx = index.search(q_vec, 3)

    chosen_idx = random.choice(idx[0])
    row = df.iloc[chosen_idx]

    # Add flavor-based variation
    ingredients = diversify_ingredients(row["ingredients"], attributes.get("flavor", "neutral"))

    return {
    "ingredients": ingredients,
    # Step 3 — extract the top score and return it
    "score": float(scores[0][0])
    }


if __name__ == "__main__":
    test_attr = {
        "product_type": "lip_balm",
        "skin_concern": "dryness",
        "texture": "moisturizing",
        "color": "clear",
        "flavor": "cherry"
    }

    print(retrieve_formulation(test_attr))
