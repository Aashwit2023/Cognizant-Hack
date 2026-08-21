# YouTube Video Scene Understanding & Context-Aware Ad Recommendation Pipeline

A multi-modal AI pipeline that ingests YouTube videos, detects scene boundaries, captions keyframes (BLIP), aligns transcript dialogue, classifies emotional tone, generates dense vector embeddings per scene, automatically syncs scene data to **Pinecone Vector Database**, and recommends context-aware advertisements with explainable rationales.

---

## 🏗️ Architecture & Pipeline Flow

```text
YouTube URL / Video ID
    │
    ▼
[1] extract_features.py (yt-dlp + youtube-transcript-api)
    ├─ Video: data/videos/<id>.mp4
    ├─ Metadata: data/metadata/<id>.json
    └─ Transcript: data/transcripts/<id>.json
    │
    ▼
[2] scene_analyzer.py (Multimodal AI Scene Pipeline)
    ├─ Step 1: PySceneDetect (Scene boundary detection + midpoint keyframes)
    ├─ Step 2: BLIP Captioning (Visual natural-language keyframe descriptions)
    ├─ Step 3: Transcript Slicing (Align dialogue to scene timestamps)
    ├─ Step 4: Emotion Classification (DistilRoBERTa emotion tagging)
    ├─ Step 5: Profile Fusion & Dense Embeddings (paraphrase-multilingual-MiniLM-L12-v2)
    └─ Step 6 (Cloud): Automatic Sync to Pinecone DB (namespace: 'video-scenes')
    │
    ▼
Outputs & Data Cache:
    ├─ data/metadata/<id>.json (Title, channel, duration, tags)
    ├─ data/transcripts/<id>.json (Timestamped Hindi/English subtitles)
    ├─ data/scenes/<id>_scenes.json (Scene timestamps, visual captions, dialogue, emotions)
    └─ data/scenes/<id>_embeddings.npy (Dense 384-dim scene vector matrix)
    │
    ▼
[3] recommend_ads_advanced.py (Hybrid Search & Scene-First Ad Engine)
    ├─ 1. Dense Semantic Retrieval (384-dim Cosine Similarity via Pinecone / Local)
    ├─ 2. Sparse Lexical Retrieval (BM25 Engine with Hindi/English tokenization)
    ├─ 3. Scene Environment & Affinity Classifier (Visual Setting, Activity, Mood)
    ├─ 4. Calibrated Hybrid Fusion & Context Consistency Gate
    └─ 5. Explainable Recommendations with Cue Point Timestamps & Natural Rationale
```

---

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd Cognizant-Hack
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
```bash
# Standard CPU:
pip install -r requirements.txt

# Or with NVIDIA GPU acceleration (e.g. RTX 3050 / 3060 / 4060):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 4. Configure Environment Variables (`.env`)
Create a `.env` file in the root directory (copy from `.env.example`):
```env
PINECONE_API_KEY=your_actual_pinecone_api_key_here
PINECONE_INDEX_NAME=ad-scene-index
```

### 5. Initialize Ad Inventory in Pinecone (One-Time Setup)
Indexes all 19 sponsor ad campaigns and target profiles in your Pinecone cloud index:
```bash
python pinecone_db.py --setup-ads
```

---

## 💻 How to Process a Fresh YouTube Video (End-to-End)

To process any brand-new YouTube video from start to finish, run:

```bash
python extract_features.py "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
```

*(You can also simply run `python extract_features.py` and paste the URL or 11-char Video ID when prompted).*

### 🔄 What Happens Automatically:
1. **Video & Metadata Download:** Downloads the `.mp4` video and video metadata using `yt-dlp` (with anti-bot headers to bypass YouTube 403 blocks).
2. **Transcript Extraction:** Fetches timestamped subtitles in Hindi (`hi`) and English (`en`).
3. **Scene Boundary Detection:** Uses `PySceneDetect` to detect visual cuts and extracts midpoint keyframes.
4. **BLIP Visual Captioning:** Generates natural language descriptions of the visual content in each scene.
5. **Emotion Analysis:** Analyzes the sentiment and mood of each scene using `DistilRoBERTa`.
6. **Dense Embeddings:** Encodes each multimodal scene into a 384-dimensional vector (`MiniLM-L12-v2`).
7. **🌲 Automatic Cloud Sync:** Scene vectors, captions, emotions, and timestamps are **automatically uploaded to Pinecone** under namespace `video-scenes`.
8. **🧹 Automatic Cleanup:** Heavy temporary `.mp4` and frame dump files are deleted to save disk space while keeping all lightweight JSON and vector data.
9. **🎯 Ad Recommendations:** The Hybrid Ad Engine executes immediately and prints the best sponsor placements with exact cue-point timestamps, relevance scores, and placement rationales.

---

## 🛠️ Other Useful Commands

### 1. Re-Run Ad Recommendations on an Already Processed Video
```bash
python recommend_ads_advanced.py <video_id>
# Example: python recommend_ads_advanced.py b3IK5GS54OI
```

### 2. Search All Indexed Video Scenes using Natural Language
You can query your entire video catalog stored in Pinecone using plain text:
```bash
# Search for outdoor bike/travel scenes
python pinecone_db.py --search "a person riding a motorcycle"

# Search for food & dining scenes
python pinecone_db.py --search "people eating food at a restaurant"

# Search for laptop/work scenes
python pinecone_db.py --search "a person working on laptop at a desk"
```

### 3. Re-Sync Local Videos to Pinecone Cloud
If you analyzed videos offline or changed your Pinecone index:
```bash
# Sync a specific video
python pinecone_db.py --sync-video <video_id>

# Sync ALL processed videos in data/scenes/ at once
python pinecone_db.py --sync-all
```

---

## 📁 Output Data Structure

All extracted data is stored in lightweight JSON and NumPy formats:

```text
data/
├── metadata/<video_id>.json       # Title, channel, duration, view count, tags
├── transcripts/<video_id>.json    # Timestamped subtitle segments
├── scenes/<video_id>_scenes.json  # Scene boundaries, BLIP captions, dialogue, emotions
└── scenes/<video_id>_embeddings.npy # 384-dimensional scene embedding matrix
```
