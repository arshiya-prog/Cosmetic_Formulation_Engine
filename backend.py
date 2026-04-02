from flask import Flask, request, jsonify
from flask_cors import CORS
import csv
import os

from modules.nlp_extractor import extract_attributes
from modules.formulation_model import retrieve_formulation
from modules.explainability import explain_ingredients
from modules.description_generator import generate_description
from modules.safety_checker import check_safety

app = Flask(__name__)
CORS(app)

ARCHIVE_PATH = "data/archive.csv"
ARCHIVE_FIELDS = ["prompt", "product_type", "primary_concern", "texture", "color",
                  "flavor", "ingredients", "description", "safe", "score"]


def _init_archive():
    """Create archive.csv with headers if it doesn't exist yet."""
    os.makedirs(os.path.dirname(ARCHIVE_PATH), exist_ok=True)
    if not os.path.exists(ARCHIVE_PATH):
        with open(ARCHIVE_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ARCHIVE_FIELDS)
            writer.writeheader()


def _save_to_archive(row: dict):
    _init_archive()
    with open(ARCHIVE_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ARCHIVE_FIELDS, extrasaction="ignore")
        writer.writerow(row)


@app.route("/")
def home():
    return "Backend is running 🚀"


@app.route("/archive", methods=["GET"])
def get_archive():
    """Return saved formulations from archive.csv (newest first, max 50)."""
    _init_archive()
    rows = []
    try:
        with open(ARCHIVE_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except FileNotFoundError:
        pass
    rows.reverse()          # newest first
    return jsonify(rows[:50])


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email', '')
    password = data.get('password', '')
    if email and len(password) >= 6:
        return jsonify({"success": True, "name": email.split("@")[0].capitalize()})
    return jsonify({"success": False, "message": "Invalid credentials."}), 401


@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.json
        user_input = data.get("prompt", "")

        if not user_input:
            return jsonify({"error": "Empty input"}), 400

        # 🔹 Extract attributes
        attributes = extract_attributes(user_input)

        # 🔹 Get formulation
        result = retrieve_formulation(attributes)
        ingredients = result.get("ingredients", "")
        score = result.get("score", 0)

        # 🔹 Explain + safety + description
        explanation = explain_ingredients(ingredients, attributes)
        safety = check_safety(ingredients, attributes)
        description = generate_description(attributes, ingredients)

        response_data = {
            "product_type": attributes.get("product_type"),
            "primary_concern": attributes.get("skin_concern"),
            "texture": attributes.get("texture"),
            "color": attributes.get("color"),
            "flavor": attributes.get("flavor", ""),
            "ingredients": ingredients,
            "benefits": explanation,
            "safe": safety,
            "description": description,
            "score": score
        }

        # ✅ Persist to archive.csv
        _save_to_archive({
            "prompt": user_input,
            "product_type": response_data["product_type"],
            "primary_concern": response_data["primary_concern"],
            "texture": response_data["texture"],
            "color": response_data["color"],
            "flavor": response_data["flavor"],
            "ingredients": ingredients,
            "description": description,
            "safe": safety,
            "score": score
        })

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/benchmark", methods=["POST"])
def benchmark():
    try:
        data = request.json
        user_ingredients = data.get("ingredients", [])
        #product_type = data.get("product_type", "").lower().replace("_", " ")
        product_type = data.get("product_type", "").lower().replace("_", " ").strip()


        competitor_db = {
            "lip balm": [
                {"name": "Carmex Classic", "ingredients": ["beeswax","camphor","menthol","salicylic acid","petrolatum"], "weaknesses": ["Contains petrolatum — not nourishing","camphor can irritate sensitive lips"]},
                {"name": "EOS Lip Balm", "ingredients": ["shea butter","jojoba oil","vitamin e","coconut oil","sunflower oil"], "weaknesses": ["No beeswax — less lasting protection","lacks skin-repair actives"]},
                {"name": "Aquaphor Lip Repair", "ingredients": ["petrolatum","shea butter","chamomile","vitamin e","panthenol"], "weaknesses": ["Heavy petrolatum base","not suitable for vegan consumers"]},
            ],
            "face cream": [
                {"name": "CeraVe Moisturizing", "ingredients": ["ceramides","hyaluronic acid","niacinamide","glycerin","petrolatum"], "weaknesses": ["Contains petrolatum — not ideal for oily skin","lacks targeted actives"]},
                {"name": "Neutrogena Hydro Boost", "ingredients": ["hyaluronic acid","glycerin","dimethicone","aloe vera","trehalose"], "weaknesses": ["Dimethicone can clog pores","no brightening or anti-aging actives"]},
                {"name": "The Ordinary Buffet", "ingredients": ["niacinamide","salicylic acid","matrixyl","argireline","hyaluronic acid"], "weaknesses": ["Watery — not suitable for dry skin","higher irritation risk"]},
            ],
            "serum": [
                {"name": "Skinceuticals C E Ferulic", "ingredients": ["vitamin c","vitamin e","ferulic acid","glycerin","hyaluronic acid"], "weaknesses": ["Very expensive","requires careful pH to stay stable"]},
                {"name": "The Ordinary Niacinamide", "ingredients": ["niacinamide","zinc","glycerin","aqua","panthenol"], "weaknesses": ["Single-concern only","can pill under makeup"]},
                {"name": "Drunk Elephant B-Hydra", "ingredients": ["niacinamide","hyaluronic acid","panthenol","pineapple extract","glycerin"], "weaknesses": ["Pineapple extract can irritate sensitive skin","premium price for basic actives"]},
            ],
            "toner": [
                {"name": "Thayers Rose Toner", "ingredients": ["rose water","witch hazel","aloe vera","glycerin","niacinamide"], "weaknesses": ["Witch hazel can over-dry sensitive skin","limited targeted actives"]},
                {"name": "Paula's Choice Toner", "ingredients": ["niacinamide","salicylic acid","green tea extract","panthenol","glycerin"], "weaknesses": ["Too strong for sensitive skin","limited hydration focus"]},
                {"name": "Klairs Supple Prep", "ingredients": ["hyaluronic acid","rose water","betaine","allantoin","glycerin"], "weaknesses": ["No actives for acne or brightening","limited concern targeting"]},
            ],
            "sunscreen": [
                {"name": "La Roche-Posay SPF 50", "ingredients": ["zinc oxide","niacinamide","thermal spring water","glycerin"], "weaknesses": ["Can leave white cast","no antioxidant protection"]},
                {"name": "EltaMD UV Clear", "ingredients": ["zinc oxide","niacinamide","lactic acid","hyaluronic acid"], "weaknesses": ["Lactic acid may cause sensitivity","higher price point"]},
                {"name": "Supergoop Unseen", "ingredients": ["red algae","frankincense","meadowfoam seed oil","vitamin e"], "weaknesses": ["Chemical filters only","not suitable for sensitive skin"]},
            ],
        }
        for key in competitor_db:
            if key in product_type or product_type in key or any(w in product_type for w in key.split()):
              matched_type = key
              break

        #matched_type = None
        #for key in competitor_db:
           # if key in product_type or product_type in key:
               # matched_type = key
               #break

        competitors = competitor_db.get(matched_type, list(competitor_db.values())[0])

        results = []
        your_set = set(i.lower().strip() for i in user_ingredients)

        for comp in competitors:
            comp_set = set(i.lower().strip() for i in comp["ingredients"])
            shared = [i for i in user_ingredients if i.lower().strip() in comp_set]
            their_only = [i for i in comp["ingredients"] if i.lower().strip() not in your_set]
            your_only = [i for i in user_ingredients if i.lower().strip() not in comp_set]
            union = your_set | comp_set
            overlap = round((len(shared) / max(len(union), 1)) * 100)

            if len(your_only) >= len(their_only) and overlap >= 40:
                verdict = "Your product leads"
            elif overlap >= 50:
                verdict = "Very similar products"
            else:
                verdict = "Different approaches"

            results.append({
                "name": comp["name"],
                "overlap": overlap,
                "shared": shared,
                "theirOnly": their_only,
                "yourOnly": your_only,
                "weaknesses": comp["weaknesses"],
                "verdict": verdict
            })

        return jsonify({
            "product_type": matched_type or "general",
            "competitors": results
        })
    
    
    

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)