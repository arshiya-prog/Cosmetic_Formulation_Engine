import json
import os

import requests

from modules.explainability import INGREDIENT_DATA
from modules.formulation_model import FLAVOR_MAP

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

MAX_INGREDIENTS = 7

# Structured output: Ollama constrains the model to this JSON schema
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "ingredients": {"type": "array", "items": {"type": "string"}},
        "description": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["ingredients", "description", "reasoning"],
}

SYSTEM_PROMPT = """You are a cosmetic formulation assistant.
Design ONE formulation for the user's request using ONLY the retrieved reference
formulations, the retrieved supporting ingredients and the verified ingredient facts provided as context.
Rules:
- Choose 4 to 7 ingredients. Every ingredient must appear in the ALLOWED INGREDIENTS list, spelled exactly as listed.
- Always include every ingredient in the REQUIRED INGREDIENTS list.
- Never use any ingredient from the AVOID list.
- Prefer ingredients whose verified benefits match the user's skin concern.
- Write a 2-3 sentence product description for the correct body area (lips, face, skin...). Only claim benefits supported by the context.
- In "reasoning", briefly explain which reference formulations and supporting ingredients you drew from and why.
Respond in JSON only."""


def _split(ings):
    return [i.strip().lower() for i in ings.split(",") if i.strip()]


def _build_context(user_input, attributes, candidates, supporting, required, avoid):
    allowed = set()
    ref_lines = []
    for n, c in enumerate(candidates, 1):
        ings = _split(c["ingredients"])
        allowed.update(ings)
        ref_lines.append(
            f"[{n}] {c['product_type']} | concern: {c['skin_concern']} | texture: {c['texture']} "
            f"| color: {c['color']} | flavor: {c['flavor']} | match score: {c['score']:.2f}\n"
            f"    ingredients: {', '.join(ings)}"
        )

    allowed.update(s["name"] for s in supporting)
    allowed.update(required)
    # Required ingredients were explicitly asked for (or are the flavor), so they stay allowed
    allowed -= (avoid - set(required))

    # Ground the model with verified facts for every allowed ingredient
    fact_lines = []
    for ing in sorted(allowed):
        data = INGREDIENT_DATA.get(ing)
        if data:
            note = f" (note: {data['note']})" if data["note"] else ""
            fact_lines.append(f"- {ing}: {', '.join(data['benefits'])}{note}")

    supporting_line = ", ".join(s["name"] for s in supporting) or "none"

    context = (
        "USER ATTRIBUTES:\n"
        f"{json.dumps(attributes)}\n\n"
        "RETRIEVED REFERENCE FORMULATIONS (best match first):\n"
        + "\n".join(ref_lines)
        + "\n\nRETRIEVED SUPPORTING INGREDIENTS (chosen for the skin concern):\n"
        + supporting_line
        + "\n\nVERIFIED INGREDIENT FACTS:\n"
        + ("\n".join(fact_lines) or "- none")
        + "\n\nALLOWED INGREDIENTS:\n" + ", ".join(sorted(allowed))
        + "\n\nREQUIRED INGREDIENTS:\n" + (", ".join(required) or "none")
        + "\n\nAVOID:\n" + (", ".join(sorted(avoid - set(required))) or "none")
        + f"\n\nUSER REQUEST:\n{user_input}"
    )
    return context, allowed


def generate_with_llm(user_input: str, attributes: dict, candidates: list,
                      supporting=(), mentioned=(), avoid=frozenset(), timeout: int = 120) -> dict:
    """Augment the prompt with retrieved formulations/ingredients and let the LLM generate the result.

    Raises on any failure so the caller can fall back to the non-LLM path.
    """
    # Required = ingredients the user named + the extract for the requested flavor
    required = list(mentioned)
    flavor_extract = FLAVOR_MAP.get(attributes.get("flavor", "neutral"))
    if flavor_extract and flavor_extract not in required:
        required.append(flavor_extract)
    avoid = set(avoid)

    context, allowed = _build_context(user_input, attributes, candidates, supporting, required, avoid)

    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ],
            "format": RESPONSE_SCHEMA,
            "stream": False,
            "options": {"temperature": 0.4},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    out = json.loads(resp.json()["message"]["content"])

    # Keep the model grounded: drop anything not in the allowed set
    ingredients = []
    for ing in out.get("ingredients", []):
        ing = ing.strip().lower()
        if ing in allowed and ing not in ingredients:
            ingredients.append(ing)
    if not ingredients:
        raise ValueError("LLM returned no valid ingredients")

    # Enforce required ingredients the model may have skipped
    for ing in required:
        if ing not in ingredients:
            ingredients.append(ing)
    if len(ingredients) > MAX_INGREDIENTS:
        optional = [i for i in ingredients if i not in required]
        keep = set(required) | set(optional[:MAX_INGREDIENTS - len(required)])
        ingredients = [i for i in ingredients if i in keep]

    return {
        "ingredients": ", ".join(ingredients),
        "description": out.get("description", "").strip(),
        "reasoning": out.get("reasoning", "").strip(),
    }
