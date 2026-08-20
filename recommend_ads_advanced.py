import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")

class AdvancedAdRecommender:
    def __init__(self, data_dir="data"):
        self.scenes_dir = os.path.join(data_dir, "scenes")
        
        print("Loading AI Model for Ad Matching...")
        self.ad_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # --- HUMARE DUMMY ADS DATABASE ---
        # Aap in ads ko apne hisaab se modify kar sakte hain
        self.ad_campaigns = {
            "Dream11 / My11Circle": "fantasy sports, cricket match, playing games, predicting score, IPL, winning money.",
            "Nike Sports Gear": "fitness, running, workout, gym, sports shoes, active lifestyle, athlete.",
            "Zomato Food Delivery": "food delivery, eating pizza burger, hungry, ordering restaurant food online, tasty meal.",
            "Apple MacBook Pro": "technology, coding, software programming, video editing, laptop, computer."
        }
        
        # Ads ke text ko vectors mein convert kar rahe hain
        self.ad_vectors = {
            ad_name: torch.tensor(self.ad_model.encode(ad_desc)) 
            for ad_name, ad_desc in self.ad_campaigns.items()
        }
        print("✅ Ads Database Ready!\n" + "="*50)

    def recommend_for_video(self, video_id):
        json_path = os.path.join(self.scenes_dir, f"{video_id}_scenes.json")
        npy_path = os.path.join(self.scenes_dir, f"{video_id}_embeddings.npy")
        
        if not os.path.exists(json_path) or not os.path.exists(npy_path):
            print(f"❌ Data for {video_id} not found. Please run the SceneAnalyzer first.")
            return

        # Load Video Data
        with open(json_path, 'r', encoding='utf-8') as f:
            video_data = json.load(f)
        
        embeddings_matrix = np.load(npy_path)
        video_tensor = torch.tensor(embeddings_matrix)

        print(f"🎬 Analyzing Video ID: {video_id}")
        print(f"Total Scenes: {video_data['total_scenes']}")
        
        # Har Ad ke liye best timestamp dhundhna
        for ad_name, ad_vector in self.ad_vectors.items():
            best_score = -1.0
            best_scene_info = None
            
            for idx, scene in enumerate(video_data['scenes']):
                # Video Scene ka vector
                scene_vector = video_tensor[idx]
                
                # AI Math: Cosine Similarity
                similarity = F.cosine_similarity(ad_vector.unsqueeze(0), scene_vector.unsqueeze(0)).item()
                
                if similarity > best_score:
                    best_score = similarity
                    best_scene_info = scene
            
            # Agar relevance 30% (0.3) se zyada hai, tabhi recommend karein
            if best_score > 0.2:
                # Timestamp format
                best_timestamp = best_scene_info['start_sec']
                mins, secs = divmod(int(best_timestamp), 60)
                time_format = f"{mins:02d}:{secs:02d}"
                
                print(f"   🎯 Recommend: [{ad_name}] at {time_format} (Relevance: {best_score*100:.1f}%)")
                print(f"      Scene Mood: {best_scene_info.get('emotion_tag', 'N/A')}")
                print(f"      Context Spoken: \"{best_scene_info.get('transcript_dialogue', 'None')[:80]}...\"")
        print("-" * 50)


# --- Run the Script ---
if __name__ == "__main__":
    recommender = AdvancedAdRecommender()
    
    # Abhi hum us video ke liye test kar rahe hain jiska data apne nikala hai
    vid_id = input("Enter Video ID to get recommendations (e.g., m9QNxYCPEDw): ")
    recommender.recommend_for_video(vid_id.strip())