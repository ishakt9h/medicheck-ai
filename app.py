from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import json
import numpy as np
import os

app = Flask(__name__)
CORS(app) 

# ── LOAD MODEL AND DATA ──────────────────────────────────────
print("🔄 Loading AI Assets...")
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open("symptom_names.json", "r") as f:
        symptom_names = json.load(f)
    print(f"✅ Ready: {len(symptom_names)} symptoms loaded.")
except Exception as e:
    print(f"❌ Error loading files: {e}")

@app.route("/")
def home():
    return send_from_directory(".", "medicheck-pro.html")

# ── SYMPTOM LIST FOR UI ──────────────────────────────────────
@app.route("/symptoms", methods=["GET"])
def get_symptoms():
    # This ensures the HTML search bar knows EXACTLY what names to use
    return jsonify({"symptoms": symptom_names})

# ── PREDICTION ENDPOINT ──────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        selected_symptoms = data.get("symptoms", [])
        
        # 1. Initialize vector with zeros
        input_vector = np.zeros(len(symptom_names), dtype=int)
        
        # 2. Match symptoms (Crucial: matching exact strings)
        found_count = 0
        for s in selected_symptoms:
            if s in symptom_names:
                idx = symptom_names.index(s)
                input_vector[idx] = 1
                found_count += 1
        
        # DEBUG PRINT: Check your terminal to see if symptoms are being "caught"
        print(f"🔍 Received: {selected_symptoms} | Matched: {found_count} symptoms")

        if found_count == 0:
            return jsonify({
                "success": True, 
                "predictions": [{"disease": "No symptoms matched our database", "confidence": "0%"}]
            })

        # 3. Predict
        proba = model.predict_proba([input_vector])[0]
        top3_idx = np.argsort(proba)[-3:][::-1]

        results = []
        for idx in top3_idx:
            conf = round(float(proba[idx]) * 100, 1)
            if conf > 0: # Only show likely diseases
                results.append({
                    "disease": le.classes_[idx],
                    "confidence": conf
                })

        return jsonify({"success": True, "predictions": results})

    except Exception as e:
        print(f"⚠️ Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)