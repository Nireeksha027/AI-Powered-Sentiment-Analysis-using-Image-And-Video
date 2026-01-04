# AI-Powered Sentiment Analysis using Image and Video

This project is a **multimodal sentiment and emotion analysis system** that detects human emotions from **images and videos** using deep learning techniques.  
It combines **computer vision, audio processing, and speech analysis** to provide accurate sentiment insights.


## 🚀 Features

### Image-Based Emotion Analysis
- Face detection using OpenCV
- Facial emotion recognition using Vision Transformer (ViT)
- Detection of **unique and duplicate faces**
- Annotated output image with emotion labels

### Video-Based Sentiment Analysis
- Facial expression analysis from video frames
- Audio-based emotion analysis using spectrograms
- Speech detection and transcription
- Voice tone and energy analysis
- Per-second emotion timeline
- Overall sentiment prediction



## 🧠 Technologies Used

- **Python**
- **Flask** – Web framework
- **PyTorch** – Deep learning framework
- **OpenCV** – Face detection
- **DeepFace** – Face verification
- **HuggingFace Transformers**
  - Vision Transformer (ViT)
  - Audio Spectrogram Transformer (AST)
- **Librosa & Torchaudio** – Audio processing
- **MoviePy** – Video processing
- **SpeechRecognition**



## 📂 Datasets Used

- **FER-2013 (Facial Expression Recognition 2013)**  
  Used for image-based facial emotion recognition.

- **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**  
  Used for video-based emotion and sentiment analysis.

> ⚠️ Due to size and licensing restrictions, datasets are **not included** in this repository.

---

## Model Management (Important)

Trained model files are **not stored in this GitHub repository** due to GitHub file size limitations.

### How the project runs without models in the repository:
- Image and audio models are automatically downloaded from **HuggingFace**
- The trained multimodal model (`best_model.pth`) is automatically downloaded from **Google Drive** using `gdown` during the first execution

📌 Please ensure you have an **active internet connection** when running the project for the first time.

---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository

git clone https://github.com/Nireeksha027/AI-Powered-Sentiment-Analysis-using-Image-And-Video.git
cd AI-Powered-Sentiment-Analysis-using-Image-And-Video

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Run the Application

python app.py

4️⃣ Open in Browser

http://127.0.0.1:5000

📁 Project Structure

<img width="460" height="282" alt="image" src="https://github.com/user-attachments/assets/f10092e4-88ad-4ef5-8cba-b2af54047d4c" />


