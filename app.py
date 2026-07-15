import streamlit as st
from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO
from datetime import datetime, timezone, timedelta
import threading
import time

st.set_page_config(page_title="Detector de Plagas", layout="wide")

st.title("🍃 Detector de Plagas en Hojas")
st.markdown("### Modelo YOLO11s - mAP50: 82.7%")

# ==========================================
# CONFIGURACIÓN DE TELEGRAM
# ==========================================
TELEGRAM_BOT_TOKEN = "8725129241:AAGBYwVLnmVfbBUa9RVjIdQD2AaOswKjinc"
TELEGRAM_CHAT_ID = "7700414080"

# Zona horaria Ecuador
ecuador_tz = timezone(timedelta(hours=-5))

# ==========================================
# INICIALIZAR SESSION STATE
# ==========================================
if 'bot_started' not in st.session_state:
    st.session_state.bot_started = False
if 'last_update_id' not in st.session_state:
    st.session_state.last_update_id = 0
if 'model' not in st.session_state:
    st.session_state.model = None
if 'CLASSES' not in st.session_state:
    st.session_state.CLASSES = ['Crítico', 'Nada Saludable', 'Saludable', 'media_saludable']

# ==========================================
# FUNCIONES DE TELEGRAM
# ==========================================
def enviar_mensaje_telegram(chat_id, texto, imagen_bytes=None):
    """Envía mensaje a Telegram"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, json={
            "chat_id": chat_id,
            "text": texto,
            "parse_mode": "Markdown"
        }, timeout=10)
        
        if imagen_bytes:
            url_foto = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            files = {'photo': imagen_bytes}
            data = {"chat_id": chat_id}
            requests.post(url_foto, files=files, data=data, timeout=10)
        
        return True
    except Exception as e:
        print(f"Error Telegram: {e}")
        return False

def descargar_imagen_telegram(file_id):
    """Descarga imagen desde Telegram"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile?file_id={file_id}"
        response = requests.get(url, timeout=10)
        file_path = response.json()['result']['file_path']
        
        url_download = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
        response = requests.get(url_download, timeout=10)
        
        return response.content
    except Exception as e:
        print(f"Error descargando imagen: {e}")
        return None

def analizar_imagen_yolo(imagen_bytes, model, CLASSES):
    """Analiza imagen con YOLO y retorna resultados"""
    try:
        image = Image.open(BytesIO(imagen_bytes))
        temp_path = "/tmp/telegram_image.jpg"
        image.save(temp_path)
        
        results = model(temp_path, verbose=False)
        boxes = results[0].boxes
        
        if len(boxes) > 0:
            mejor = max(boxes, key=lambda b: float(b.conf[0]))
            clase = CLASSES[int(mejor.cls[0])]
            conf = float(mejor.conf[0]) * 100
            
            result_img = results[0].plot()
            img_byte_arr = BytesIO()
            Image.fromarray(result_img).save(img_byte_arr, format='JPEG')
            
            return clase, conf, img_byte_arr.getvalue()
        else:
            return None, 0, None
    except Exception as e:
        print(f"Error analizando imagen: {e}")
        return None, 0, None

# ==========================================
# BOT DE TELEGRAM (THREAD ÚNICO)
# ==========================================
def bot_telegram_polling():
    """Revisa mensajes nuevos de Telegram cada 3 segundos"""
    while True:
        try:
            offset = st.session_state.last_update_id + 1 if st.session_state.last_update_id > 0 else -1
            
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates?offset={offset}&timeout=10"
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                updates = response.json().get('result', [])
                
                for update in updates:
                    update_id = update.get('update_id')
                    message = update.get('message', {})
                    chat_id = message.get('chat', {}).get('id')
                    
                    # Actualizar último update procesado
                    if update_id > st.session_state.last_update_id:
                        st.session_state.last_update_id = update_id
                    
                    # Verificar si es una foto
                    if 'photo' in message:
                        photo = message['photo'][-1]
                        file_id = photo['file_id']
                        
                        # Enviar mensaje de "procesando"
                        enviar_mensaje_telegram(chat_id, "🔍 Analizando imagen...")
                        
                        # Descargar y analizar
                        imagen_bytes = descargar_imagen_telegram(file_id)
                        if imagen_bytes:
                            clase, conf, imagen_resultado = analizar_imagen_yolo(
                                imagen_bytes, 
                                st.session_state.model, 
                                st.session_state.CLASSES
                            )
                            
                            ahora = datetime.now(ecuador_tz)
                            
                            if clase:
                                respuesta = f"""
🍃 *Resultado del Análisis*

🎯 *Clase:* {clase}
📊 *Confianza:* {conf:.2f}%
⏰ *Hora:* {ahora.strftime('%H:%M:%S')}
 *Fecha:* {ahora.strftime('%d/%m/%Y')}

{'🚨 *¡ALERTA!* Revisa la planta inmediatamente' if clase in ['Crítico', 'Nada Saludable'] else '✅ Hoja en buen estado'}
                                """
                                enviar_mensaje_telegram(chat_id, respuesta, imagen_resultado)
                            else:
                                enviar_mensaje_telegram(chat_id, "❌ No se detectó ninguna hoja")
        
        except Exception as e:
            print(f"Error en polling: {e}")
        
        time.sleep(3)

# ==========================================
# CARGAR MODELO (una sola vez)
# ==========================================
if st.session_state.model is None:
    try:
        with st.spinner("Cargando modelo..."):
            model_url = "https://huggingface.co/EAMB2001/detector-plagas-modelo/resolve/main/modelo.pt"
            response = requests.get(model_url)
            model_path = "/tmp/modelo.pt"
            with open(model_path, "wb") as f:
                f.write(response.content)
            st.session_state.model = YOLO(model_path)
        st.success("✅ Modelo cargado")
    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")

# ==========================================
# INICIAR BOT (solo una vez)
# ==========================================
if not st.session_state.bot_started and st.session_state.model is not None:
    st.session_state.bot_started = True
    threading.Thread(target=bot_telegram_polling, daemon=True).start()
    st.sidebar.success("🤖 Bot de Telegram activo")

# ==========================================
# INTERFAZ WEB
# ==========================================
st.sidebar.markdown("### 📱 Uso del Bot de Telegram")
st.sidebar.info("""
1. Abre Telegram
2. Busca: @detector_plagas_alertas_bot
3. Envía una imagen de hoja
4. Recibe el análisis automáticamente
""")

uploaded_file = st.file_uploader("📷 Sube una imagen de hoja", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None and st.session_state.model is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(uploaded_file, caption="Imagen original", use_column_width=True)
    
    if st.button("🔍 Analizar Hoja"):
        with st.spinner("Analizando imagen..."):
            image = Image.open(uploaded_file)
            temp_path = "/tmp/temp_image.jpg"
            image.save(temp_path)
            
            results = st.session_state.model(temp_path, verbose=False)
            boxes = results[0].boxes
            
            with col2:
                if len(boxes) > 0:
                    mejor = max(boxes, key=lambda b: float(b.conf[0]))
                    clase = st.session_state.CLASSES[int(mejor.cls[0])]
                    conf = float(mejor.conf[0]) * 100
                    
                    st.success(f"✅ **{clase}**")
                    st.metric("Confianza", f"{conf:.2f}%")
                    
                    result_img = results[0].plot()
                    st.image(result_img, caption="Resultado", use_column_width=True)
                    
                    st.write("###  Todas las detecciones:")
                    for box in boxes:
                        cls = st.session_state.CLASSES[int(box.cls[0])]
                        confidence = float(box.conf[0]) * 100
                        st.write(f"• **{cls}**: {confidence:.2f}%")
                    
                    # Alerta para casos críticos
                    if clase in ['Crítico', 'Nada Saludable']:
                        img_byte_arr = BytesIO()
                        image.save(img_byte_arr, format='JPEG')
                        
                        ahora = datetime.now(ecuador_tz)
                        mensaje = f"""
🚨 *ALERTA DE PLAGA DETECTADA*

🍃 *Clase:* {clase}
📊 *Confianza:* {conf:.2f}%
⏰ *Hora:* {ahora.strftime('%H:%M:%S')}
📅 *Fecha:* {ahora.strftime('%d/%m/%Y')}

⚠️ *Acción recomendada:* Revisar planta inmediatamente
                        """
                        
                        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
                        requests.post(url, json={
                            "chat_id": TELEGRAM_CHAT_ID,
                            "text": mensaje,
                            "parse_mode": "Markdown"
                        }, timeout=10)
                        
                        st.warning("⚠️ **Alerta enviada a Telegram**")
                    else:
                        st.info("✅ Detección normal - Sin alerta")
                else:
                    st.warning("No se detectó ninguna hoja")

st.markdown("---")
st.markdown("""
### ℹ️ Información:
- **Alertas automáticas:** Se envían cuando se detecta 'Crítico' o 'Nada Saludable'
- **Bot de Telegram:** Envía imágenes al bot para análisis instantáneo
- **Modelo:** YOLO11s entrenado con mAP50: 82.7%
- **Zona horaria:** Ecuador (UTC-5)
""")
