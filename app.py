import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import json
import pandas as pd
import altair as alt 

IMG_SIZE = 224  

with open("class_names.json") as f:
    class_names = json.load(f)

model = tf.keras.models.load_model("best_vgg19.keras")

def preprocess_image(image, img_size=IMG_SIZE):
    img = image.convert("RGB")
    img = img.resize((img_size, img_size))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

st.title(" Brain Tumor MRI Classifier")

uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI", use_container_width=True)

    processed_img = preprocess_image(image)

    preds = model.predict(processed_img)
    pred_idx = np.argmax(preds, axis=1)[0]
    pred_label = class_names[pred_idx]

    st.subheader(f"✅ Predicted Class: {pred_label}")
    st.write("Model confidence for each class:")

    probs = preds[0]
    prob_df = pd.DataFrame({
        'Class': class_names,
        'Probability': probs
    })

    st.dataframe(prob_df.style.format({'Probability': '{:.2%}'}))

    chart = alt.Chart(prob_df).mark_bar().encode(
        x=alt.X('Class', sort=None),
        y=alt.Y('Probability', scale=alt.Scale(domain=[0, 1])),
        tooltip=['Class', alt.Tooltip('Probability', format='.2%')],
        color=alt.Color('Class', legend=None)
    ).properties(width=500, height=300)

    st.altair_chart(chart, use_container_width=True)
