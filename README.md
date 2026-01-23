# 🍽 Food Waste Impact Tracker (Vision + Nutrition + Sustainability)

### 🔗 Live Demo  
👉 https://huggingface.co/spaces/samikshachougule-hub/Food_Waste_Tracker

---

## 🧾 Overview

Food waste is a major global sustainability challenge, contributing to carbon emissions, excessive water usage, and economic loss.  
This project detects leftover food from user-uploaded images and estimates its environmental impact (calories wasted, CO₂ emissions, and water footprint).  
It also suggests simple reuse ideas to minimize waste.

---

## 🎯 Problem Statement

A significant portion of leftover food ends up being thrown away due to lack of awareness about:

- Nutritional value
- Environmental impact
- Reuse possibilities

This project addresses the gap by providing impact metrics and reuse suggestions through a simple web interface.

---

## 🚀 Features

✔ Vision-based food classification (Indian food dataset)  
✔ Calorie waste estimation  
✔ CO₂ footprint estimation (per kg)  
✔ Water footprint estimation (per kg)  
✔ Rule-based reuse suggestions  
✔ Web deployment (HuggingFace Spaces)  
✔ Lightweight MobileNetV2 for fast inference  
✔ Works on CPU & browser  
✔ No user installation required  

---

## 🗂 System Architecture

User Upload → Preprocessing → CNN Classifier (MobileNetV2)
→ Calories / CO₂ / Water Estimation
→ Reuse Suggestions → UI Display

---

## 📚 Dataset

**Dataset Used:**  
📦 *Indian Food Classification* (20 classes)  
Source: Kaggle (By Pushkar Jain)

**Dataset Stats**

| Property | Value |
|---|---|
| Classes | 20 |
| Images | ~5000 |
| Format | JPG |
| Type | Multiclass |
| Domain | Indian Cuisine |

---

## 🤖 Model

**Model:** `MobileNetV2 (Transfer Learning)`  
**Training Pipeline:**

- Stage 1: Freeze base → 10 epochs
- Stage 2: Fine-tune → 5 epochs
- Final val accuracy: **~60%+**

**Export Format:** TensorFlow `SavedModel`

mobilenet_food_classifier/
├── saved_model.pb
├── variables/
├── assets/
└── fingerprint.pb

**Class labels stored in:** `label_map.json`

---

## 📊 Environmental Impact Estimation

Metrics used:

| Metric | Data Source |
|---|---|
| Calories | Standard nutritional tables |
| CO₂ footprint | Food lifecycle emission estimates |
| Water footprint | Water usage per kg |

Example values:

burger → 5.3 kg CO₂/kg, 2350 L/kg
chai → 0.8 kg CO₂/kg, 800 L/kg
pizza → 3.5 kg CO₂/kg, 1500 L/kg

---

## ♻ Reuse Suggestions (Rule-based)

Example mapping:

samosa → samosa chaat, crushed samosa wrap
chapati → chapati rolls, chapati upma, chapati chips
rice → fried rice, kheer, rice cutlets

Stored in: `utils/reuse_rules.json`

---

## 🧰 Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Model | TensorFlow + MobileNetV2 |
| Backend | Python |
| Dataset | Kaggle |
| Deployment | HuggingFace Spaces |
| Format | SavedModel |
| Training | Google Colab |

---

## 🗂 Project Structure

Food_Waste_Tracker/
├── app.py
├── model/
├── utils/
│ ├── calorie_table.csv
│ ├── co2_table.csv
│ ├── water_table.csv
│ └── reuse_rules.json
├── requirements.txt
└── README.md

---

🌱 Use Cases

✔ Sustainability Education
✔ College Projects & Workshops
✔ Environmental Awareness Campaigns
✔ Food Tech Apps
✔ Smart Waste Management Systems

🧩 Future Enhancements

🔜 Possible improvements:

Serving size estimation

Multi-food plate detection

OCR-based expiry detection

LLM-based reuse recipe generation

IoT smart fridge integration

User history analytics

Carbon pricing calculation

🏆 Acknowledgements

Dataset by: Pushkar Jain
Frameworks: TensorFlow, Streamlit
Deployment: HuggingFace Spaces

📄 License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).  
This means you may use, modify, and share the project for non-commercial purposes, with attribution.

👤 Author

Name: Samiksha Chougule Patil
Role: Developer & ML Engineer
Project Type: Internship + College Submission


---


---
