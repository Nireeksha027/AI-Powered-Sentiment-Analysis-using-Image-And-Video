import os, re, cv2, numpy as np, torch, librosa, speech_recognition as sr
import torch.nn as nn
import torchaudio
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from moviepy.editor import VideoFileClip
from torchvision import transforms, models
from transformers import ASTModel, AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
from deepface import DeepFace
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import gdown



# ─── Flask setup ───
app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True

UPLOAD_FOLDER, FACE_FOLDER, VIDEO_FOLDER, AUDIO_FOLDER = "uploads", "faces", "videos", "audio"
for fld in (UPLOAD_FOLDER, FACE_FOLDER, VIDEO_FOLDER, AUDIO_FOLDER):
    os.makedirs(fld, exist_ok=True)

# ─── IMAGE PIPELINE ───
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load pretrained high-accuracy ViT model (91.77% from Alpiyildo)
print("🔹 Loading ViT model (Alpiyildo/vit-Facial-Expression-Recognition)...")
model_name = "Alpiyildo/vit-Facial-Expression-Recognition"
processor = AutoImageProcessor.from_pretrained(model_name)
vit_img_model = AutoModelForImageClassification.from_pretrained(model_name)
vit_img_model.eval()
print("✅ ViT model loaded successfully!")

# ─── FACE ANALYSIS ───
def detect_faces(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.05, 3, minSize=(20, 20))
    crops = []
    for i, (x, y, w, h) in enumerate(faces):
        margin = 10
        crop = img[max(0, y - margin):y + h + margin, max(0, x - margin):x + w + margin]
        fpath = os.path.join(FACE_FOLDER, f"face_{i}.jpg")
        cv2.imwrite(fpath, crop)
        crops.append((fpath, (x, y, w, h)))
    return crops, img

def analyze_faces(img_path):
    crops, full = detect_faces(img_path)
    data, uniq, dups = [], [], []

    for fp, (x, y, w, h) in crops:
        dup = False
        # --- Check duplicate faces using DeepFace ---
        for s in uniq:
            try:
                if DeepFace.verify(fp, s['path'], model_name="Facenet", enforce_detection=False)["verified"]:
                    dup = True
                    dups.append(s)
                    break
            except Exception:
                pass
        if not dup:
            uniq.append({"path": fp})

        # --- Emotion detection using ViT model ---
        try:
            img = Image.open(fp).convert("RGB")
            inputs = processor(images=img, return_tensors="pt")
            with torch.no_grad():
                outputs = vit_img_model(**inputs)
                logits = outputs.logits
                pred_idx = logits.argmax(-1).item()
                emotion = vit_img_model.config.id2label[pred_idx]
        except Exception as e:
            emotion = "Unknown"
            print(f"[ViT Error] {e}")

        data.append({
            "image_path": f"/faces/{os.path.basename(fp)}",
            "dominant_emotion": emotion
        })

        # Draw rectangle and emotion label on original image
        cv2.rectangle(full, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(full, emotion, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    ann_path = os.path.join(UPLOAD_FOLDER, "annotated.jpg")
    cv2.imwrite(ann_path, full)
    return data, len(uniq), len(dups), f"/uploads/{os.path.basename(ann_path)}"

@app.route("/analyze_image", methods=["POST"])
def analyze_image():
    if "file" not in request.files:
        return "No file part"
    f = request.files["file"]
    if f.filename == "":
        return "No selected file"
    path = os.path.join(UPLOAD_FOLDER, secure_filename(f.filename))
    f.save(path)

    faces, u, d, annot = analyze_faces(path)
    return render_template("result.html", analyzed_data=faces, unique_faces=u, duplicate_faces=d, annotated_image=annot)


# ─── VIDEO PIPELINE (unchanged) ───
vid_labels = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

class MultiModalClassifier(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.heads.head = nn.Identity()
        self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.fusion = nn.Sequential(
            nn.Linear(768 + 768, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x_img, x_spec):
        zi = self.vit(x_img)
        za = self.ast(input_values=x_spec).pooler_output
        return self.fusion(torch.cat([zi, za], dim=1))

# Load both models
MODEL_PATH = "models/best_model.pth"

if not os.path.exists(MODEL_PATH):
    os.makedirs("models", exist_ok=True)
    print("Downloading model file (one-time, ~660MB)...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

best_model = MultiModalClassifier()
best_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"), strict=False)
best_model.eval()

mel, to_db = MelSpectrogram(16000, n_fft=1024, hop_length=512, n_mels=128), AmplitudeToDB()
v_trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def logmel_from_wav(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    spec = to_db(mel(wav))
    spec = (spec - spec.mean()) / (spec.std() + 1e-9)
    if spec.shape[-1] < 1024:
        spec = torch.nn.functional.pad(spec, (0, 1024 - spec.shape[-1]))
    spec = spec[..., :1024].squeeze(0).T.contiguous()
    return spec

@app.route("/analyze_video", methods=["POST"])
def analyze_video():
    model_type = request.form.get("model_type", "best")
    model = best_model if model_type == "best" else best_model

    video_file = request.files.get("video")
    vpath, fname = None, None

    if video_file and video_file.filename != "":
        fname = secure_filename(video_file.filename)
        vpath = os.path.join(VIDEO_FOLDER, fname)
        video_file.save(vpath)
    else:
        return "❌ No video file provided"

    try:
        clip = VideoFileClip(vpath).subclip(0, min(30, VideoFileClip(vpath).duration))
        if not clip.audio:
            return "❌ The video has no audio."

        apath = os.path.join(AUDIO_FOLDER, "full_audio.wav")
        clip.audio.write_audiofile(apath, fps=16000, codec="pcm_s16le", logger=None)

        rec = sr.Recognizer()
        with sr.AudioFile(apath) as src:
            aud = rec.record(src)
        try:
            transcription = rec.recognize_google(aud)
            talking = "Yes"
        except Exception:
            transcription = ""
            talking = "No"

        y, _ = librosa.load(apath, sr=16000)
        f0 = librosa.yin(y, fmin=50, fmax=300)
        tone = "High Energy" if np.nanmean(f0) > 170 else "Low Energy"

        per_second, counts = [], {e: 0 for e in vid_labels}
        duration = int(clip.duration)
        for t in range(duration):
            try:
                frame = Image.fromarray(clip.get_frame(t)).convert("RGB")
                xi = v_trans(frame).unsqueeze(0)
                chunk = os.path.join(AUDIO_FOLDER, f"chunk_{t}.wav")
                clip.audio.subclip(max(0, t - 0.5), min(clip.duration, t + 0.5)).write_audiofile(chunk, fps=16000, codec="pcm_s16le", logger=None)
                if os.path.getsize(chunk) < 4096:
                    os.remove(chunk)
                    continue
                xs = logmel_from_wav(chunk).unsqueeze(0)
                os.remove(chunk)
                with torch.no_grad():
                    pred = model(xi, xs).argmax(1).item()
                emo = vid_labels[pred]
                per_second.append((t, emo))
                counts[emo] += 1
            except Exception as e:
                print(f"[ERR] {t}s {e}")

        if not per_second:
            return "❌ Emotion model failed on this video."

        overall = max(counts, key=counts.get).capitalize()
        return render_template("video_result.html",
                               video_filename=fname,
                               transcription=transcription,
                               tone=tone,
                               talking_status=talking,
                               per_second_emotions=per_second,
                               emotion_frequency=counts,
                               sentiment=overall,
                               expression=overall,
                               source_type="Uploaded File")
    except Exception as e:
        print(f"[PROCESSING ERROR] {e}")
        return "❌ Unexpected error during video processing."

# ─── Static ───
@app.route("/uploads/<f>")
def _u(f): return send_from_directory(UPLOAD_FOLDER, f)

@app.route("/faces/<f>")
def _f(f): return send_from_directory(FACE_FOLDER, f)

@app.route("/videos/<filename>")
def video_file(filename): return send_from_directory(VIDEO_FOLDER, filename)

@app.route("/")
def home(): return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=True, extra_files=["templates/video_result.html"])
