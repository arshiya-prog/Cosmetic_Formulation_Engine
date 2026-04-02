def generate_description(attributes, ingredients):

    product = attributes.get("product_type", "").replace("_", " ").title()
    concern = attributes.get("skin_concern", "")
    flavor = attributes.get("flavor", "neutral")

    base = f"This {flavor} flavored {product.lower()} is specially formulated"

    if concern == "dryness":
        base += " to deeply hydrate and repair dry lips"
    elif concern == "sensitive":
        base += " to gently soothe and protect sensitive lips"
    elif concern == "brightening":
        base += " to enhance natural lip tone and radiance"
    else:
        base += " to improve overall lip health"

    base += ". "

    base += "It contains a blend of nourishing ingredients like "
    base += ", ".join(ingredients.split(",")[:3])
    base += ", ensuring effective care and protection."

    return base