# MediCheck AI 🩺

> AI-powered symptom checker trained on 246,000 real medical cases.  
> Predicts the most likely disease from a set of symptoms - +80% accuracy across 721 diseases.

---

## What it does

You give it a list of symptoms. It returns the top 3 most likely diseases with a confidence percentage in under 1 second.

```
Enter symptoms: fever, cough, nasal congestion, headache

Top 3 predictions:
  1. acute sinusitis          89.20%
  2. rhinitis                 61.40%
  3. common cold              44.10%
```

---

## Dataset

| Property | Value |
|---|---|
| File | `Final_Augmented_dataset_Diseases_and_Symptoms.csv` |
| Rows | 246,945 |
| Symptoms (features) | 377 |
| Diseases (classes) | 773 (721 after filtering rare ones) |
| Format | Binary matrix each row is a patient, each column is a symptom (0 or 1) |


---

## Model

| Property | Value |
|---|---|
| Algorithm | Random Forest |
| Trees | 200 |
| Max depth | 30 |
| Features per split | sqrt(377) ≈ 19 |
| Class balancing | `class_weight="balanced"` |
| Training time | ~2 minutes (12 cores) |
| Test accuracy | **+80%** on 49,389 rows |
| Target accuracy | 85%+ (v2) |

### Top symptoms by importance

| Symptom | Importance |
|---|---|
| cough | 2.00% |
| shortness of breath | 1.89% |
| headache | 1.38% |
| sharp abdominal pain | 1.33% |
| emotional symptoms | 1.24% |

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/ishak-t9h/medicheck-ai.git
cd medicheck-ai
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add the dataset**
```
data/
└── Final_Augmented_dataset_Diseases_and_Symptoms.csv
```

---

## Usage

### Train the model
```bash
python train_model.py
```
This creates 3 files:
- `model.pkl` — the trained Random Forest
- `label_encoder.pkl` — maps numbers back to disease names
- `symptom_names.json` — list of all 377 symptoms


```




## Roadmap

- [x] v1 — Random Forest baseline (+80% accuracy)
- [ ] v2 — Tune hyperparameters, target 85%+
- [ ] v3 — Add patient age & sex as features
- [ ] v4 — REST API (Flask) for integration
- [ ] v5 — Arabic & French symptom input support

---

## Disclaimer

This model is for **informational and educational purposes only**.  
It is not a medical device and does not replace professional medical advice.  
Always consult a licensed doctor for diagnosis and treatment.

---

## License

free to use, modify, and distribute with attribution.
