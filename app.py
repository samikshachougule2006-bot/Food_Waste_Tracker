import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import json

# ---------------------- Load Model ---------------------- #
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/mobilenet_indian_food.keras")

model = load_model()

# ---------------------- Load Classes ---------------------- #
with open("model/classes.json", "r") as f:
    class_names = json.load(f)

# ---------------------- Load Tables ---------------------- #
calories = pd.read_csv("utils/calorie_table.csv")
co2 = pd.read_csv("utils/co2_table.csv")
water = pd.read_csv("utils/water_table.csv")

with open("utils/reuse_rules.json") as f:
    reuse_rules = json.load(f)

# ---------------------- UI ---------------------- #
st.title("🍽️ Food Waste Impact Tracker")
st.write("Upload a food image and view calorie + environmental impact + reuse suggestions.")

uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # preprocess
    img_resized = img.resize((224,224))
    arr = np.array(img_resized) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # prediction
    pred = model.predict(arr)
    class_idx = np.argmax(pred)
    food_label = class_names[class_idx]

    st.subheader(f"🍛 Detected Food: **{food_label}**")

    # ----- Calories -----
    cal = calories[calories.food == food_label].calories_per_serving.values
    if len(cal):
        st.write(f"🔥 **Calories:** {cal[0]} kcal per serving")

    # ----- CO2 -----
    c = co2[co2.food == food_label].co2_kg_per_kg.values
    if len(c):
        st.write(f"🌍 **CO₂ Footprint:** {c[0]} kg CO₂ per kg")

    # ----- Water -----
    w = water[water.food == food_label].water_liters_per_kg.values
    if len(w):
        st.write(f"💧 **Water Use:** {w[0]} L per kg")

    # ----- Reuse Suggestions -----
    if food_label in reuse_rules:
        st.write("♻ **Reuse Ideas:**")
        for idea in reuse_rules[food_label]:
            st.write(f"- {idea}")
    else:
        st.write("♻ No reuse suggestions available for this item.")