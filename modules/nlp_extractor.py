from transformers import pipeline

# Load zero-shot classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


def extract_flavor(text):
    flavors = ["cherry", "blueberry", "strawberry", "mint", "vanilla"]
    for f in flavors:
        if f in text.lower():
            return f
    return "neutral"


def extract_skin_concern(text):
    text = text.lower()
    if "dry" in text:
        return "dryness"
    elif "sensitive" in text:
        return "sensitive"
    elif "acne" in text:
        return "acne"
    elif "bright" in text:
        return "brightening"
    elif "oil" in text:
        return "oily_skin"
    return "general"


def classify(text, labels):
    result = classifier(text, labels)
    return result["labels"][0]


def extract_attributes(user_input: str) -> dict:

    product_types = ["lip balm", "face cream", "serum", "toner", "sunscreen"]
    textures = ["moisturizing", "lightweight", "rich", "gel", "watery"]
    colors = ["rose", "clear", "white", "yellow"]

    return {
        "product_type": classify(user_input, product_types).replace(" ", "_"),
        "skin_concern": extract_skin_concern(user_input),
        "texture": classify(user_input, textures),
        "color": classify(user_input, colors),
        "flavor": extract_flavor(user_input)
    }
