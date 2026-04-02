def check_safety(ingredients, attributes):

    warnings = []

    skin = attributes.get("skin_concern", "")

    if "peppermint oil" in ingredients and skin == "sensitive":
        warnings.append("Peppermint oil may irritate sensitive lips")

    if "fragrance" in ingredients:
        warnings.append("Contains fragrance which may cause irritation")

    if not warnings:
        return "Safe for use ✅"

    return "⚠️ " + " | ".join(warnings)
