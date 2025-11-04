import uvicorn
import pickle
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Inisialisasi Aplikasi FastAPI
app = FastAPI(title="API Klasifikasi Teks Judi", version="1.0")

# 2. Tentukan Model Input (menggunakan Pydantic)
# Ini akan menjadi format JSON yang diterima API Anda
class TextIn(BaseModel):
    text: str

# 3. Muat Model dan Tokenizer saat startup
# Pastikan file-file ini ada di folder yang sama dengan main.py
try:
    model = tf.keras.models.load_model('modelLajuA (8).keras')
    
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    # Jika gagal, API tetap berjalan tapi endpoint akan error
    model = None
    tokenizer = None

# 4. Definisikan Konstanta dari Notebook Anda
MAX_LENGTH = 700
PADDING_TYPE = 'post'
TRUNC_TYPE = 'post'

# 5. Buat Endpoint Prediksi
@app.post("/predict")
def predict_text(item: TextIn):
    """
    Menerima teks dan mengembalikan prediksi 'Judi Online' atau 'Bukan Judi Online'.
    """
    if not model or not tokenizer:
        return {"error": "Model atau tokenizer tidak berhasil dimuat."}

    try:
        # Ambil teks dari request
        new_text = [item.text]

        # Lakukan preprocessing yang SAMA PERSIS seperti di notebook
        new_seq = tokenizer.texts_to_sequences(new_text)
        new_pad = pad_sequences(new_seq, 
                                maxlen=MAX_LENGTH, 
                                padding=PADDING_TYPE, 
                                truncating=TRUNC_TYPE)

        # Lakukan prediksi
        prediction = model.predict(new_pad)
        
        # Dapatkan nilai probabilitas (cth: [[0.9815]])
        probability = prediction[0][0] 
        
        # Tentukan label berdasarkan ambang batas 0.5
        if probability > 0.5:
            label = "Judi Online"
        else:
            label = "Bukan Judi Online"

        # Kembalikan hasil dalam format JSON
        return {
            "text": item.text,
            "prediction_label": label,
            "prediction_probability": float(probability) 
        }

    except Exception as e:
        return {"error": f"Terjadi kesalahan saat prediksi: {str(e)}"}

# 6. (Opsional) Jalankan server jika file ini dieksekusi langsung
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
