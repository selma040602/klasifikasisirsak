import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import cv2

# ------------ Definisi FixedDropout ------------
class FixedDropout(tf.keras.layers.Dropout):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super().__init__(rate, noise_shape, seed, **kwargs)

    def call(self, inputs, training=None):
        return super().call(inputs, training=training)


# ------------ Fungsi Load Model ------------
@st.cache_resource
def load_model(model_name):
    return tf.keras.models.load_model(
        model_name,
        custom_objects={
            "swish": tf.keras.activations.swish,
            "FixedDropout": FixedDropout
        }
    )


# ------------ FUNGSI ZOOM ------------
def zoom_image(img, zoom_level, center_x=None, center_y=None):
    """
    Zoom pada gambar dengan faktor zoom tertentu
    zoom_level: 1.0 = no zoom, 2.0 = 2x zoom, dst
    center_x, center_y: koordinat pusat zoom (0-1), None = center
    """
    if zoom_level <= 1.0:
        return img
    
    width, height = img.size
    
    # Default center jika tidak ditentukan
    if center_x is None:
        center_x = 0.5
    if center_y is None:
        center_y = 0.5
    
    # Hitung ukuran crop berdasarkan zoom level
    crop_width = width / zoom_level
    crop_height = height / zoom_level
    
    # Hitung koordinat crop
    left = max(0, int(center_x * width - crop_width / 2))
    top = max(0, int(center_y * height - crop_height / 2))
    right = min(width, int(center_x * width + crop_width / 2))
    bottom = min(height, int(center_y * height + crop_height / 2))
    
    # Crop dan resize kembali ke ukuran original
    cropped = img.crop((left, top, right, bottom))
    zoomed = cropped.resize((width, height), Image.Resampling.LANCZOS)
    
    return zoomed


def auto_zoom_to_object(img, zoom_factor=1.8):
    """
    Auto zoom ke objek dengan deteksi edge/kontras
    """
    # Convert ke array untuk processing
    img_array = np.array(img)
    
    # Convert ke grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply threshold untuk detect object
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Ambil contour terbesar
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Hitung center dari object
        center_x = (x + w/2) / img.width
        center_y = (y + h/2) / img.height
        
        # Zoom ke center object
        return zoom_image(img, zoom_factor, center_x, center_y)
    
    # Jika tidak ada object terdeteksi, zoom ke center
    return zoom_image(img, zoom_factor)


# ---------------- FUNGSI PREDIKSI ----------------
def predict_image(img, model):
    # Tampilkan gambar original
    st.image(img, caption="Gambar Input", use_container_width=True)
    
    # Apply zoom otomatis (tanpa preprocessing)
    img_zoomed = auto_zoom_to_object(img, zoom_factor=1.8)
    
    # Resize untuk model
    img_size = model.input_shape[1:3]
    img_resized = img_zoomed.resize(img_size, Image.Resampling.LANCZOS)
    img_array = np.array(img_resized).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # ------------ Prediksi + Hitung Waktu ------------
    start_time = time.time()
    pred = model.predict(img_array)
    end_time = time.time()
    
    inference_time_ms = (end_time - start_time) * 1000
    st.write(f"⏱️ Waktu inferensi: {inference_time_ms:.2f} ms")
    
    # ------------ Hasil Prediksi ------------
    if pred.ndim == 1 or pred.shape[1] == 1:  # sigmoid
        prob = float(pred[0] if pred.ndim == 1 else pred[0][0])
        if prob > 0.5:
            st.success(f"✅ Hasil: **Segar** 🌶 (Probabilitas: {prob*100:.2f}%)")
        else:
            st.error(f"❌ Hasil: **Busuk** 🤢 (Probabilitas: {(1-prob)*100:.2f}%)")
    else:  # softmax
        label = ["Busuk", "Segar"]
        kelas = np.argmax(pred, axis=1)[0]
        prob_busuk = pred[0][0] * 100
        prob_segar = pred[0][1] * 100
        
        if kelas == 1:
            st.success(f"✅ Hasil: **{label[kelas]}** 🌶")
        else:
            st.error(f"❌ Hasil: **{label[kelas]}** 🤢")
        
        # Tampilkan detail probabilitas
        st.write(f"📊 Detail Probabilitas:")
        st.write(f"- Busuk: {prob_busuk:.2f}%")
        st.write(f"- Segar: {prob_segar:.2f}%")


# ------------ UI Streamlit ------------
st.title("🌶 Klasifikasi Cabe Jamu: Segar atau Busuk")
st.write("**Zoom Version** - Auto-zoom ke objek cabai")

# 🔹 Dropdown untuk pilih model
model_choice = st.selectbox(
    "Pilih model yang akan digunakan:",
    ["adam_fold2 (1).h5", "model_cnn_aug.h5", "sgd_fold5.h5"]
)

# Load model sesuai pilihan
model = load_model(model_choice)

st.write("---")

# ---------------- UPLOAD GAMBAR ----------------
st.subheader("📂 Upload Gambar Cabai")
uploaded_file = st.file_uploader("Upload gambar cabe...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    predict_image(img, model)

st.write("---")

# ---------------- KAMERA HP (REAL-TIME) ----------------
st.subheader("📷 Ambil Gambar dari Kamera HP")
camera_image = st.camera_input("Gunakan kamera HP untuk foto cabai")

if camera_image is not None:
    img = Image.open(camera_image).convert("RGB")
    predict_image(img, model)

st.write("---")
st.info("""
💡 **Tips untuk hasil terbaik:**
1. Gunakan pencahayaan yang baik dan merata
2. Fokus pada objek cabai, hindari background yang ramai
3. Ambil foto dari jarak yang cukup dekat
4. Pastikan cabai terlihat jelas, tidak blur
5. Sistem secara otomatis akan mendeteksi dan zoom ke objek cabai
""")