# 🎯 Context-Aware Advertising (AI-Powered Recommendation System)

## 📌 Project Overview
Advertising effectiveness increasingly depends on relevance and timing. Traditional ad placement relies heavily on broad audience segmentation. This project introduces a **Context-Aware Advertising System** that analyzes video scenes, transcripts, and emotional tones to place ads at the most relevant timestamps. 

For example: 
* A fitness ad is recommended during energetic workout scenes.
* A food delivery ad is recommended when food or hunger is discussed.
* Multilingual support ensures relevance even if the dialogue is in Hindi/Hinglish.

## 🚀 Key Features
1. **Automated Data Extraction:** Downloads YouTube videos, metadata, and transcripts (supports legacy and modern API structures).
2. **Smart Scene Detection:** Uses `PySceneDetect` to divide videos into logical scenes rather than blind time-intervals.
3. **Multi-Modal AI Analysis:**
   * **Visuals:** Uses Salesforce's `BLIP` model to generate captions for video frames.
   * **Emotions:** Uses `DistilRoBERTa` to classify the emotional tone of the scene.
   * **Dialogue:** Aligns YouTube transcripts with exact scene timestamps.
4. **Multilingual Context Matching:** Uses `paraphrase-multilingual-MiniLM-L12-v2` to create embeddings that understand both English and Hindi context.
5. **Ad Recommendation Engine:** Calculates **Cosine Similarity** between Ad Campaigns and Video Scenes to find the perfect timestamp for ad insertion.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **AI & Deep Learning:** PyTorch, HuggingFace Transformers, Sentence-Transformers
* **Computer Vision:** OpenCV, PySceneDetect, Pillow
* **Data Extraction:** yt-dlp, youtube-transcript-api
* **Data Processing:** NumPy, Pandas, JSON

## 📂 Folder Structure
```text
project/
│
├── data/
│   ├── frames/         # Extracted video frames
│   ├── metadata/       # Video details (likes, views, description)
│   ├── scenes/         # .json profiles and .npy embedding vectors
│   ├── transcripts/    # Downloaded subtitle/caption files
│   └── videos/         # Downloaded MP4 files
│
├── extract_features.py       # Step 1: Downloads video & extracts baseline features
├── scene_analyzer.py         # Step 2: Generates captions, emotions & vector embeddings
├── recommend_ads_advanced.py # Step 3: Matches ads to scenes using Cosine Similarity
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation