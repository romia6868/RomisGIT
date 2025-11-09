import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# 🧠 טוען את המודל מהדרייב שלך או מהתיקייה של גיטהאב
# אם תעלי את המודל לגיטהאב (לקובץ באותה תיקייה), תשני כאן רק את השם
model = tf.keras.models.load_model("cnn_flowers_model.keras")

# 🏷️ שמות הקטגוריות
class_names = ["Daisy", "Dandelion", "Tulip"]

# 🎨 עיצוב בסיסי של האפליקציה
st.set_page_config(page_title="🌷 Flower Classifier", page_icon="🌸", layout="centered")
st.title("🌸 Flower Classifier App")
st.write("העלו תמונה של פרח כדי לזהות את סוגו 🌼")

# 📸 העלאת תמונה
uploaded_file = st.file_uploader("בחרי תמונה (JPG או PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="תמונה שהועלתה", use_column_width=True)

    # 🔄 עיבוד תמונה לפני חיזוי
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 🔍 חיזוי
    if st.button("סווג את התמונה"):
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

        st.success(f"המודל מזהה: **{predicted_class}** 🌸")
        st.write(f"✅ רמת ביטחון: {confidence:.2f}")
