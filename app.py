import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2
import pytesseract
import re

st.set_page_config(page_title="AI Analyzer", layout="wide")
st.title("🧠 Мой ИИ Анализатор")
st.write("Загрузите изображение, и я попробую его распознать.")

# --- Загрузка моделей ---
@st.cache_resource
def load_mnist_model():
    # Загружаем сохраненную ранее модель (mnist_model.h5 должна лежать в репозитории)
    return keras.models.load_model('mnist_model.h5')

# --- Блок для распознавания цифр ---
def predict_digit(image):
    model = load_mnist_model()
    # Предобработка под MNIST (28x28, оттенки серого)
    img = image.convert('L').resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28)
    
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    return predicted_digit, confidence

# --- Блок для анализа карты (Tesseract) ---
def analyze_map(image):
    # Конвертируем PIL в формат OpenCV
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Увеличиваем для лучшего распознавания
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
    
    # Ищем текст
    config = r'--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789.,'
    text = pytesseract.image_to_string(thresh, config=config)
    
    numbers = re.findall(r'\b\d{2,3}(?:[.,]\d)?\b', text)
    heights = []
    for num in numbers:
        val = float(num.replace(',', '.'))
        if 40.0 <= val <= 450.0:
            heights.append(val)
    return list(set(heights))

# --- Интерфейс ---
uploaded_file = st.file_uploader("Выберите изображение...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Загруженное изображение', use_column_width=True)
    
    # Выбор режима
    mode = st.radio("Что сделать с картинкой?", ("Распознать рукописную цифру", "Проанализировать топографическую карту"))
    
    if st.button("Запустить ИИ"):
        with st.spinner('Думаю...'):
            if mode == "Распознать рукописную цифру":
                try:
                    digit, conf = predict_digit(image)
                    st.success(f"**Результат:** Это цифра **{digit}** (уверенность: {conf:.1f}%)")
                except Exception as e:
                    st.error(f"Ошибка при загрузке модели. Убедитесь, что файл mnist_model.h5 лежит в репозитории. Ошибка: {e}")
            
            elif mode == "Проанализировать топографическую карту":
                # ВАЖНО: Для работы Tesseract на Streamlit Cloud нужен файл packages.txt
                try:
                    heights = analyze_map(image)
                    if heights:
                        st.success(f"**Найдено высот:** {len(heights)}")
                        st.write(f"Список: {heights}")
                        st.write(f"Мин: {min(heights)} м, Макс: {max(heights)} м, Среднее: {np.mean(heights):.1f} м")
                    else:
                        st.warning("Не удалось распознать высоты на этой карте.")
                except Exception as e:
                    st.error(f"Ошибка OCR: {e}")
