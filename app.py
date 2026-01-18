import streamlit as st
st.set_page_config(page_title="Food Waste Impact Tracker", layout="centered")

import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import json

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    model = tf.saved_model.load("model/mobilenet_food_classifier")
    infer = model.signatures["serving_default"]
    return infer

infer = load_model()

# ---------------- Load Label Map ----------------
with open("model/label_map.json", "r") as f:
    CLASS_NAMES = json.load(f)

# ---------------- Load Tables ----------------
calorie = pd.read_csv("utils/calorie_table.csv")      # column: calories_per_serving
co2 = pd.read_csv("utils/co2_table.csv")              # column: co2_kg_per_kg
water = pd.read_csv("utils/water_table.csv")          # column: water_l_per_kg

with open("utils/reuse_rules.json", "r") as f:
    reuse_rules = json.load(f)

# ---------------- UI ----------------
st.title("🍽 Food Waste Impact Tracker")
st.write("Upload a food image to estimate calorie waste, CO₂ footprint, water usage, and reuse suggestions.")

uploaded = st.file_uploader("Upload food image", type=["jpg","jpeg","png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", width=300)

    # ---- Preprocess (Matching Training) ----
    img_resized = img.resize((224, 224))
    arr = np.array(img_resized).astype("float32")
    arr = np.expand_dims(arr, axis=0)

    # Inference
    outputs = infer(tf.constant(arr))
    logits = list(outputs.values())[0].numpy()
    idx = int(np.argmax(logits))
    conf = float(np.max(logits))
    food = CLASS_NAMES[idx]

    st.success(f"Predicted food: **{food}** ({conf:.2f} confidence)")

    # ---------------- Nutrition Impact ----------------
    row = {}

    # Calories
    if food in calorie.food.values:
        row["calories"] = float(calorie.loc[calorie.food == food, "calories_per_serving"].values[0])
    else:
        row["calories"] = None

    # CO2
    if food in co2.food.values:
        row["co2"] = float(co2.loc[co2.food == food, "co2_kg_per_kg"].values[0])
    else:
        row["co2"] = None

    # Water
    if food in water.food.values:
        row["water"] = float(water.loc[water.food == food, "water_l_per_kg"].values[0])
    else:
        row["water"] = None

    # Display Environmental Impact
    st.subheader("📊 Environmental Impact")
    st.write(f"🔥 **Calories:** `{row['calories']} kcal`")
    st.write(f"🌍 **CO₂ footprint:** `{row['co2']} kg per kg`")
    st.write(f"💧 **Water footprint:** `{row['water']} L per kg`")

    # ---------------- Reuse Suggestions ----------------
    st.subheader("♻ Reuse Suggestions")
    if food in reuse_rules:
        for s in reuse_rules[food]:
            st.write(f"• {s}")
    else:
        st.write("No reuse suggestions available for this item yet.")