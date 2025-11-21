# 🚨 Toxic Comment Detection (Binary Classification)

This project is an end-to-end **Toxic vs Non-Toxic Comment Classification System** built using:

- **Deep Learning (CNN / LSTM model)**
- **FastAPI** for backend inference
- **Streamlit** for frontend UI
- **Docker** for containerized deployment

The system classifies any user comment into:

- 🟥 **Toxic**
- 🟩 **Non-Toxic**

---

## 📌 Project Architecture

```
User → Streamlit UI → FastAPI API → Toxicity Model → Prediction
```

### Components:
- **frontend/** – Streamlit user interface  
- **fastapp/** – FastAPI backend with model inference  
- **model/** – tokenizer + saved deep learning model  
- **docker-compose.yml** – runs both apps together  

---

## 🚀 Features

### ✔ Binary Toxic Comment Classification  
Predicts:
- **0 → Non-Toxic**  
- **1 → Toxic**

### ✔ FastAPI Backend  
- `/predict` endpoint  
- Returns prediction + confidence  
- Handles tokenization, padding, inference  

### ✔ Streamlit Frontend  
- Clean textbox for entering comments  
- Color-coded output  
- Real-time classification  

### ✔ Dockerized Microservices  
- Run using `docker-compose up`  
- Streamlit + FastAPI communicate internally  

---

## 🔧 How to Run the Project

### 1️⃣ Clone Repository
```bash
git clone <repo-url>
cd Toxic-Comment-Classifier
```

---

### 2️⃣ Run With Docker (Recommended)

```bash
docker-compose up --build
```

### Access:
- 🌐 **Streamlit UI:** http://localhost:8501  
- 🧪 **FastAPI Docs (Swagger):** http://localhost:8000/docs  

---

### 3️⃣ Run Without Docker (Local Mode)

#### Start FastAPI:
```bash

uvicorn fastapp.main:app --reload 
```

#### Start Streamlit:
```bash
cd frontend
streamlit run frontend/frontend.py
```

---

## 🧠 Model Details

This project uses a binary deep learning classifier.

### Pipeline:
- Text cleaning  
- Tokenization  
- Sequence padding  
- Embedding layer  
- CNN 
- Dense layer with **sigmoid output**  

### Output:
- `0` → Non-Toxic  
- `1` → Toxic  

---

## 📡 API Example

### Request:
```json
{
  "text": "You are so stupid!"
}
```

### Response:
```json
{
  "prediction": "Toxic",
  "confidence": 0.91
}
```

---

## 🎨 Streamlit UI

- Simple text input  
- Classification result with confidence score
- User-friendly layout  

---

## 📦 Tech Stack

### Backend:
- FastAPI  
- Uvicorn  
- TensorFlow / Keras  
- Pickle  

### Frontend:
- Streamlit  
- Requests  

### DevOps:
- Docker  
- Docker Compose  

---

## 🛠 Future Enhancements
- Add multi-label toxicity detection  
- Add explanation (LIME / SHAP)   

---

