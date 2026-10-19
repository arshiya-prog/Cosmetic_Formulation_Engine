# Ingredients to keep out of generated formulas, per skin concern
AVOID_ALWAYS = ["fragrance"]
AVOID_BY_CONCERN = {
    "sensitive": ["peppermint oil", "witch hazel", "glycolic acid", "retinol"],
    "dryness": ["witch hazel", "salicylic acid"],
}


def ingredients_to_avoid(attributes):
    return set(AVOID_ALWAYS + AVOID_BY_CONCERN.get(attributes.get("skin_concern", ""), []))


def check_safety(ingredients, attributes):

    warnings = []

    skin = attributes.get("skin_concern", "")
    present = [i.strip().lower() for i in ingredients.split(",")]

    if "peppermint oil" in present and skin == "sensitive":
        warnings.append("Peppermint oil may irritate sensitive lips")

    if "fragrance" in present:
        warnings.append("Contains fragrance which may cause irritation")

    for ing in AVOID_BY_CONCERN.get(skin, []):
        if ing in present and ing != "peppermint oil":
            warnings.append(f"{ing.title()} is not recommended for {skin.replace('_', ' ')} skin")

    if not warnings:
        return "Safe for use ✅"

    return "⚠️ " + " | ".join(warnings)
