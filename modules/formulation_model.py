import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
import random

from modules.explainability import INGREDIENT_DATA

model = SentenceTransformer("all-MiniLM-L6-v2")

FLAVOR_MAP = {
    "cherry": "cherry extract",
    "blueberry": "blueberry extract",
    "strawberry": "strawberry extract",
    "mint": "mint extract",
    "vanilla": "vanilla extract"
}


DATA_PATH = "data/formulations.csv"

# Weights for hybrid re-ranking (sum to 1.0, so the final score stays in 0-1)
W_SEMANTIC = 0.45   # cosine similarity of embeddings
W_TYPE = 0.25       # same product type
W_CONCERN = 0.20    # same skin concern
W_TEXTURE = 0.05
W_FLAVOR = 0.05

# Concerns that are close enough to earn half credit in re-ranking
RELATED_CONCERNS = {
    "acne": ["oily_skin"],
    "oily_skin": ["acne"],
    "dryness": ["hydration"],
    "hydration": ["dryness"],
}

# Concern described in the same vocabulary as the ingredient benefits, for ingredient retrieval
CONCERN_BENEFITS = {
    "dryness": "deep hydration, repairs dryness, locks moisture, prevents moisture loss",
    "hydration": "hydrates skin, draws moisture, intense hydration",
    "sensitive": "soothes irritation, calms redness, anti-inflammatory, gentle",
    "acne": "treats acne, unclogs pores, controls oil, reduces inflammation",
    "oily_skin": "controls oil, reduces pores, balances oil, unclogs pores",
    "brightening": "brightens skin, evens skin tone, antioxidant",
    "anti_aging": "reduces wrinkles, boosts collagen, improves elasticity, reduces signs of aging",
    "protection": "UV protection, broad-spectrum protection, prevents sunburn",
}

_cache = {"mtime": None, "df": None, "index": None}


def _readable(value):
    return str(value).replace("_", " ")


def load_data():
    df = pd.read_csv(DATA_PATH).dropna(subset=["product_type"]).reset_index(drop=True)

    # Ensure flavor column exists
    if "flavor" not in df.columns:
        df["flavor"] = "neutral"
    df["flavor"] = df["flavor"].fillna("neutral")

    # Convert structured data to a descriptive sentence (ingredients included so
    # ingredient words in the prompt, e.g. "vitamin c", also influence similarity)
    df["text"] = df.apply(
        lambda r: f"{_readable(r['product_type'])} for {_readable(r['skin_concern'])} skin, "
                  f"{r['texture']} texture, {r['color']} color, {r['flavor']} flavor, "
                  f"with {r['ingredients']}",
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


def _get_index():
    """Build the dataset + FAISS index once, and rebuild only if the CSV changes."""
    mtime = os.path.getmtime(DATA_PATH)
    if _cache["mtime"] != mtime:
        df = load_data()
        _cache.update(mtime=mtime, df=df, index=build_index(df))
    return _cache["df"], _cache["index"]


def _flavor_match(row_flavor, wanted):
    if wanted in ("", "neutral"):
        return 1.0 if row_flavor == "neutral" else 0.5
    return 1.0 if row_flavor == wanted else 0.0


def diversify_ingredients(ingredients_str, flavor):
    ingredients = [i.strip() for i in ingredients_str.split(",")]

    if flavor in FLAVOR_MAP:
        ingredients.append(FLAVOR_MAP[flavor])

    return ", ".join(list(set(ingredients)))


def retrieve_candidates(attributes: dict, k: int = 3, user_input: str = ""):
    """Return the top-k formulations (the 'R' in RAG).

    Stage 1: FAISS cosine similarity over all formulations.
    Stage 2: hybrid re-ranking that also rewards matching product type, skin concern,
             texture and flavor, so e.g. a 'sensitive' request is not outranked by a
             row that only shares the flavor.
    """
    df, index = _get_index()

    # Build query text in the same sentence style as the documents, plus the raw prompt
    query = (f"{_readable(attributes.get('product_type',''))} for "
             f"{_readable(attributes.get('skin_concern',''))} skin, "
             f"{attributes.get('texture','')} texture, {attributes.get('color','')} color, "
             f"{attributes.get('flavor','')} flavor. {user_input}")

    # Normalize query to unit vector. Now query is a point in the same 384D space
    q_vec = model.encode([query], convert_to_numpy=True)
    q_vec = q_vec / np.linalg.norm(q_vec)

    # Cosine similarity against every row (dataset is small, so search all of it)
    sims, idx = index.search(q_vec, len(df))

    candidates = []
    for sim, i in zip(sims[0], idx[0]):
        row = df.iloc[i]
        type_match = 1.0 if row["product_type"] == attributes.get("product_type") else 0.0
        wanted_concern = attributes.get("skin_concern")
        if row["skin_concern"] == wanted_concern:
            concern_match = 1.0
        elif row["skin_concern"] in RELATED_CONCERNS.get(wanted_concern, []):
            concern_match = 0.5
        else:
            concern_match = 0.0
        texture_match = 1.0 if row["texture"] == attributes.get("texture") else 0.0
        flavor_match = _flavor_match(row["flavor"], attributes.get("flavor", "neutral"))

        score = (W_SEMANTIC * max(float(sim), 0.0) + W_TYPE * type_match +
                 W_CONCERN * concern_match + W_TEXTURE * texture_match +
                 W_FLAVOR * flavor_match)

        candidates.append({
            "product_type": row["product_type"],
            "skin_concern": row["skin_concern"],
            "texture": row["texture"],
            "color": row["color"],
            "flavor": row["flavor"],
            "ingredients": row["ingredients"],
            "similarity": round(float(sim), 4),
            "score": round(score, 4),
        })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:k]


_ing_cache = {"names": None, "vecs": None}


def _ingredient_index():
    """Embed every ingredient in the knowledge base by its name + benefits (built once)."""
    if _ing_cache["names"] is None:
        names = list(INGREDIENT_DATA.keys())
        texts = [f"{n}: {', '.join(INGREDIENT_DATA[n]['benefits'])}; "
                 f"{', '.join(INGREDIENT_DATA[n]['verification'])}" for n in names]
        vecs = model.encode(texts, convert_to_numpy=True)
        _ing_cache.update(names=names, vecs=vecs / np.linalg.norm(vecs, axis=1, keepdims=True))
    return _ing_cache["names"], _ing_cache["vecs"]


def known_ingredients():
    """All ingredient names the system knows (knowledge base + dataset)."""
    df, _ = _get_index()
    names = set(INGREDIENT_DATA.keys())
    for ings in df["ingredients"]:
        names.update(i.strip().lower() for i in ings.split(",") if i.strip())
    return names


def find_mentioned_ingredients(user_input: str):
    """Ingredients the user explicitly named in the prompt, e.g. 'with vitamin c'."""
    text = f" {user_input.lower()} "
    found = [n for n in known_ingredients() if f" {n} " in text or f" {n}," in text]
    # Prefer longer names ("aloe vera extract" over "aloe vera")
    found.sort(key=len, reverse=True)
    result = []
    for n in found:
        if not any(n in longer for longer in result):
            result.append(n)
    return result


def retrieve_ingredients(attributes: dict, user_input: str = "", k: int = 4, exclude=()):
    """Retrieve individual ingredients whose verified benefits fit the concern.

    This fills gaps in the formulation dataset, e.g. there is no brightening toner,
    but niacinamide / vitamin c are retrieved from the knowledge base by their benefits.
    """
    names, vecs = _ingredient_index()
    concern = attributes.get("skin_concern", "")
    query = CONCERN_BENEFITS.get(concern, f"{_readable(concern)} skin care, nourishes skin")
    q = model.encode([query], convert_to_numpy=True)[0]
    q = q / np.linalg.norm(q)
    sims = vecs @ q

    flavor_extracts = set(FLAVOR_MAP.values()) | {"strawberry extract", "blueberry extract", "vanilla extract"}
    ranked = sorted(zip(names, sims), key=lambda x: x[1], reverse=True)
    out = []
    for name, sim in ranked:
        if name in exclude or name in flavor_extracts:
            continue
        out.append({"name": name, "similarity": round(float(sim), 4)})
        if len(out) == k:
            break
    return out


def retrieve_formulation(attributes: dict, candidates=None):
    """Non-LLM fallback: pick one of the top-3 matches at random."""
    if candidates is None:
        candidates = retrieve_candidates(attributes, 3)

    chosen = random.choice(candidates)

    # Add flavor-based variation
    ingredients = diversify_ingredients(chosen["ingredients"], attributes.get("flavor", "neutral"))

    return {
    "ingredients": ingredients,
    # Step 3 — extract the top score and return it
    "score": candidates[0]["score"]
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
