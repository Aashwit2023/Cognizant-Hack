# YouTube Video Scene Understanding & Feature Extraction Pipeline

A multi-modal AI pipeline that ingests YouTube videos, detects scene boundaries, captions keyframes (BLIP), aligns transcript dialogue, classifies emotional tone, and generates dense vector embeddings per scene.

---

## 🏗️ Pipeline Architecture (Steps 1 to 5)

```text
YouTube URL
    │
    ▼
[1] extract_features.py (yt-dlp + youtube-transcript-api)
    │  ├─ Video: data/videos/<id>.mp4
    │  ├─ Metadata: data/metadata/<id>.json
    │  └─ Transcript: data/transcripts/<id>.json
    │
    ▼
[2] scene_analyzer.py
    │  ├─ Step 1: PySceneDetect (Scene boundary detection + keyframes)
    │  ├─ Step 2: BLIP Captioning (Visual natural-language descriptions)
    │  ├─ Step 3: Transcript Slicing (Align dialogue to scene timestamps)
    │  ├─ Step 4: Emotion Classification (DistilRoBERTa emotion tagging)
    │  └─ Step 5: Scene Profile Fusion & Dense Embeddings (all-MiniLM-L6-v2)
    │
    ▼
Outputs:
  ├─ data/scenes/<id>/scene_XXX.jpg (Midpoint keyframe images)
  ├─ data/scenes/<id>_scenes.json (Multimodal scene metadata & profiles)
  └─ data/scenes/<id>_embeddings.npy (Dense 384-dim scene vector matrix)
```

---

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd Cognizant
```

### 2. Create and activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

#### Standard CPU:
```bash
pip install -r requirements.txt
```

#### NVIDIA GPU (e.g. RTX 3050 / 3060 / 4060):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## 💻 How to Run

### 1. Run End-to-End Extraction & Scene Analysis
```bash
python extract_features.py
# Enter any YouTube URL when prompted
```

### 2. Run Scene Analysis on an Already Downloaded Video
```bash
python scene_analyzer.py <video_id>
# Example: python scene_analyzer.py xLTCivIB4kU
```
