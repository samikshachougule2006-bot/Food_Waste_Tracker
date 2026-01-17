import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import json

def load_model():
    model = tf.keras.models.load_model("model/mobilenet_indian_food.keras")
    return model

model = load_model()

with open("model/classes.json") as f:
    class_names = json.load(f)

calories = pd.read_csv("utils/calorie_table.csv")
co2 = pd.read_csv("utils/co2_table.csv")
water = pd.read_csv("utils/water_table.csv")

with open("utils/reuse_rules.json") as f:
    reuse_rules = json.load(f)

def main():
    st.title("🍽 Food Waste Impact Tracker")

    uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, use_column_width=True)

        img_resized = img.resize((224,224))
        arr = np.expand_dims(np.array(img_resized)/255.0, axis=0)

        pred = model.predict(arr)
        label = class_names[int(np.argmax(pred))]

        st.subheader(f"Detected: {label}")

        cal = calories[calories.food==label].calories_per_serving.values
        if len(cal): st.write(f"🔥 Calories: {cal[0]} kcal")

        c = co2[co2.food==label].co2_kg_per_kg.values
        if len(c): st.write(f"🌍 CO₂: {c[0]} kg CO₂ / kg")

        w = water[water.food==label].water_liters_per_kg.values
        if len(w): st.write(f"💧 Water: {w[0]} L / kg")

        if label in reuse_rules:
            st.write("♻ Reuse Ideas:")
            for idea in reuse_rules[label]:
                st.write(f"- {idea}")

if __name__ == "__main__":
    main()
