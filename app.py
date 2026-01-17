import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import json

# Load model
def load_model():
    return tf.keras.models.load_model("model/mobilenet_indian_food.keras")

model = load_model()

# Load class labels
with open("model/classes.json") as f:
    class_names = json.load(f)

# Load tables
calories = pd.read_csv("utils/calorie_table.csv")
co2 = pd.read_csv("utils/co2_table.csv")
water = pd.read_csv("utils/water_table.csv")

# Load reuse suggestions
with open("utils/reuse_rules.json") as f:
    reuse_rules = json.load(f)

def main():
    st.title("🍽 Food Waste Impact Tracker 🇮🇳")
    st.write("Upload a food image to estimate wasted calories, CO₂, water consumption and reuse suggestions.")

    uploaded_file = st.file_uploader("Upload a food image", type=["jpg","jpeg","png"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        img_resized = img.resize((224,224))
        arr = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

        pred = model.predict(arr)
        idx = int(np.argmax(pred))
        label = class_names[idx]

        st.subheader(f"🍛 Detected Food: **{label}**")

        # Calories
        cal = calories[calories.food == label].calories_per_serving.values
        if len(cal):
            st.write(f"🔥 Calories (per serving): **{cal[0]} kcal**")

        # CO2
        c = co2[co2.food == label].co2_kg_per_kg.values
        if len(c):
            st.write(f"🌍 Carbon Footprint: **{c[0]} kg CO₂ / kg**")

        # Water
        w = water[water.food == label].water_liters_per_kg.values
        if len(w):
            st.write(f"💧 Water Usage: **{w[0]} L / kg**")

        # Reuse ideas
        if label in reuse_rules:
            st.write("♻ Reuse Suggestions:")
            for idea in reuse_rules[label]:
                st.write(f"• {idea}")
        else:
            st.write("♻ No reuse suggestions available.")

if __name__ == "__main__":
    main()
