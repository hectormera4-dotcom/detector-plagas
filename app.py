import streamlit as st
from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO
import os

st.set_page_config(page_title="Detector de Plagas", layout="wide")

st.title("🍃 Detector de Plaga Mosca Blanca en Hojas de Algodon")
st.markdown("### By: Erick Mera-Kevin Garcia")

# Cargar modelo
@st.cache_resource
def load_model():
    model_url = "https://huggingface.co/EAMB2001/detector-plagas-modelo/resolve/main/modelo.pt"
    
    response = requests.get(model_url)
    model_path = "/tmp/modelo.pt"
    with open(model_path, "wb") as f:
        f.write(response.content)
    
    return YOLO(model_path)

try:
    model = load_model()
    CLASSES = ['Crítico', 'Nada Saludable', 'Saludable', 'media_saludable']
except Exception as e:
    st.error(f"Error cargando el modelo: {e}")
    model = None

# Subir imagen
uploaded_file = st.file_uploader("📷 Sube una imagen de hoja", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None and model is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(uploaded_file, caption="Imagen original", use_column_width=True)
    
    if st.button("🔍 Analizar"):
        with st.spinner("Analizando imagen..."):
            image = Image.open(uploaded_file)
            temp_path = "/tmp/temp_image.jpg"
            image.save(temp_path)
            
            results = model(temp_path, verbose=False)
            boxes = results[0].boxes
            
            with col2:
                if len(boxes) > 0:
                    mejor = max(boxes, key=lambda b: float(b.conf[0]))
                    clase = CLASSES[int(mejor.cls[0])]
                    conf = float(mejor.conf[0]) * 100
                    
                    st.success(f"✅ **{clase}**")
                    st.metric("Confianza", f"{conf:.2f}%")
                    
                    result_img = results[0].plot()
                    st.image(result_img, caption="Resultado", use_column_width=True)
                    
                    st.write("### 📊 Todas las detecciones:")
                    for box in boxes:
                        cls = CLASSES[int(box.cls[0])]
                        confidence = float(box.conf[0]) * 100
                        st.write(f"• **{cls}**: {confidence:.2f}%")
                else:
                    st.warning("No se detectó ninguna hoja")
