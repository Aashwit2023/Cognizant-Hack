import os
import sys
import json
import time
import re
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# UTF-8 output encoding support
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


class PineconeManager:
    """
    Pinecone Vector Database Manager for YouTube Scene Understanding & Ad Placement.
    Stores and indexes:
    - 384-dim Video Scene Embeddings (namespace: 'video-scenes')
    - 384-dim Ad Inventory Campaign Embeddings (namespace: 'ad-inventory')
    """
    DEFAULT_INDEX_NAME = "ad-scene-index"
    DIMENSION = 384  # Matches paraphrase-multilingual-MiniLM-L12-v2
    METRIC = "cosine"

    def __init__(self, api_key=None, index_name=None):
        self.api_key = api_key or os.getenv("PINECONE_API_KEY", "").strip()
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", self.DEFAULT_INDEX_NAME).strip()
        self.pc = None
        self.index = None
        self._embedding_model = None

        if self.api_key and self.api_key != "your_pinecone_api_key_here":
            try:
                from pinecone import Pinecone
                self.pc = Pinecone(api_key=self.api_key)
                self.index = self._get_or_create_index()
            except Exception as e:
                print(f"⚠️ Pinecone initialization warning: {e}")
                self.pc = None
                self.index = None
        else:
            self.pc = None
            self.index = None

    def is_available(self):
        """Returns True if connected to Pinecone and the index is ready."""
        return self.index is not None

    def _get_embedding_model(self):
        """Loads and caches the SentenceTransformer model on demand."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)
        return self._embedding_model

    def _get_or_create_index(self):
        """Ensures the Pinecone serverless index exists and returns the Index instance."""
        if not self.pc:
            return None
        from pinecone import ServerlessSpec

        try:
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            if self.index_name not in existing_indexes:
                print(f"🌲 Auto-creating Pinecone Serverless Index '{self.index_name}' (dim={self.DIMENSION}, metric={self.METRIC})...")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.DIMENSION,
                    metric=self.METRIC,
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                # Wait until index is ready
                while not self.pc.describe_index(self.index_name).status['ready']:
                    time.sleep(1)
                print(f"✅ Pinecone index '{self.index_name}' created and ready!")
            
            return self.pc.Index(self.index_name)
        except Exception as e:
            print(f"❌ Error getting/creating Pinecone index '{self.index_name}': {e}")
            return None

    # =========================================================================
    # 1. VIDEO SCENES UPSERT
    # =========================================================================
    def upsert_video_scenes(self, video_id, scenes_data, embeddings_matrix, namespace="video-scenes", batch_size=50):
        """
        Uploads scene vectors and rich metadata for an analyzed video into Pinecone.
        """
        if not self.is_available():
            print("ℹ️ Pinecone API key not configured. Skipping cloud vector upsert (local vectors preserved).")
            return False

        if embeddings_matrix is None or len(embeddings_matrix) == 0:
            print(f"⚠️ No embeddings to upload for video '{video_id}'.")
            return False

        scenes = scenes_data.get("scenes", []) if isinstance(scenes_data, dict) else scenes_data
        video_metadata = scenes_data.get("metadata", {}) if isinstance(scenes_data, dict) else {}
        video_title = str(video_metadata.get("title", f"Video {video_id}"))[:200]
        channel = str(video_metadata.get("channel", "Unknown"))[:100]

        vectors_to_upsert = []
        for idx, scene in enumerate(scenes):
            if idx >= len(embeddings_matrix):
                break

            vector = embeddings_matrix[idx].tolist() if hasattr(embeddings_matrix[idx], "tolist") else list(embeddings_matrix[idx])
            scene_id = scene.get("scene_id", idx + 1)
            vector_id = f"{video_id}#scene_{scene_id:03d}"

            # Safely truncate text fields to stay well under Pinecone's 40KB metadata limit
            caption = str(scene.get("visual_caption", ""))[:500]
            dialogue = str(scene.get("transcript_dialogue", ""))[:1000]
            emotion = str(scene.get("emotion_tag", "neutral"))[:50]
            start_sec = float(scene.get("start_sec", 0.0))
            end_sec = float(scene.get("end_sec", 0.0))
            duration = round(end_sec - start_sec, 2)

            metadata = {
                "type": "scene",
                "video_id": str(video_id),
                "video_title": video_title,
                "channel": channel,
                "scene_id": int(scene_id),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": duration,
                "visual_caption": caption,
                "transcript_dialogue": dialogue,
                "emotion_tag": emotion
            }

            vectors_to_upsert.append({
                "id": vector_id,
                "values": vector,
                "metadata": metadata
            })

        # Batch upsert into Pinecone
        try:
            total_uploaded = 0
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                self.index.upsert(vectors=batch, namespace=namespace)
                total_uploaded += len(batch)
            print(f"🌲 [Pinecone] Successfully upserted {total_uploaded} scene vectors for video '{video_id}' (ns: '{namespace}')!")
            return True
        except Exception as e:
            print(f"❌ [Pinecone] Upsert failed for video '{video_id}': {e}")
            return False

    # =========================================================================
    # 2. AD CAMPAIGNS UPSERT
    # =========================================================================
    def upsert_ad_campaigns(self, campaigns_dict, namespace="ad-inventory"):
        """
        Indexes ad sponsor campaign profiles and embeddings in Pinecone.
        """
        if not self.is_available():
            print("ℹ️ Pinecone API key not configured. Cannot upsert ad campaigns to cloud.")
            return False

        model = self._get_embedding_model()
        vectors_to_upsert = []

        print(f"🌲 [Pinecone] Indexing {len(campaigns_dict)} ad campaigns into namespace '{namespace}'...")
        for ad_name, data in campaigns_dict.items():
            enriched_desc = f"{data['semantic_profile']} Keywords: {data.get('keywords', '')}"
            vec = model.encode(enriched_desc, normalize_embeddings=True).tolist()
            
            slug_id = re.sub(r'[^a-zA-Z0-9_-]', '_', ad_name).lower().strip('_')
            
            metadata = {
                "type": "ad",
                "ad_name": ad_name,
                "brand": str(data.get("brand", "")),
                "category": str(data.get("category", "")),
                "keywords": str(data.get("keywords", ""))[:500],
                "semantic_profile": str(data.get("semantic_profile", ""))[:800],
                "target_environments": data.get("target_environments", []),
                "preferred_emotions": data.get("preferred_emotions", []),
            }

            vectors_to_upsert.append({
                "id": f"ad_{slug_id}",
                "values": vec,
                "metadata": metadata
            })

        try:
            self.index.upsert(vectors=vectors_to_upsert, namespace=namespace)
            print(f"✅ [Pinecone] Successfully indexed {len(vectors_to_upsert)} ad campaigns!")
            return True
        except Exception as e:
            print(f"❌ [Pinecone] Failed to upsert ad campaigns: {e}")
            return False

    # =========================================================================
    # 3. VECTOR RETRIEVAL / QUERYING
    # =========================================================================
    def query_ads_for_scene(self, scene_vector, top_k=5, filter_dict=None, namespace="ad-inventory"):
        """
        Finds the top matching sponsor ads for a given scene vector.
        """
        if not self.is_available():
            return []

        try:
            vec = scene_vector.tolist() if hasattr(scene_vector, "tolist") else list(scene_vector)
            response = self.index.query(
                vector=vec,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict,
                namespace=namespace
            )
            results = []
            for match in response.get("matches", []):
                results.append({
                    "id": match["id"],
                    "score": round(float(match["score"]), 4),
                    "metadata": match.get("metadata", {})
                })
            return results
        except Exception as e:
            print(f"⚠️ [Pinecone] Ad query failed: {e}")
            return []

    def query_scenes_for_ad(self, ad_vector, top_k=10, filter_dict=None, namespace="video-scenes"):
        """
        Finds the top matching video scenes across all indexed videos for an ad.
        """
        if not self.is_available():
            return []

        try:
            vec = ad_vector.tolist() if hasattr(ad_vector, "tolist") else list(ad_vector)
            response = self.index.query(
                vector=vec,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict,
                namespace=namespace
            )
            results = []
            for match in response.get("matches", []):
                results.append({
                    "id": match["id"],
                    "score": round(float(match["score"]), 4),
                    "metadata": match.get("metadata", {})
                })
            return results
        except Exception as e:
            print(f"⚠️ [Pinecone] Scene query failed: {e}")
            return []

    def search_video_scenes(self, query_text, top_k=5, filter_dict=None, namespace="video-scenes"):
        """
        Performs natural language semantic search across all indexed video scenes.
        """
        if not self.is_available():
            print("❌ Pinecone not available. Please set PINECONE_API_KEY in .env.")
            return []

        model = self._get_embedding_model()
        vec = model.encode(query_text, normalize_embeddings=True).tolist()
        return self.query_scenes_for_ad(vec, top_k=top_k, filter_dict=filter_dict, namespace=namespace)


# =============================================================================
# CLI UTILITY
# =============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pinecone Vector Database Manager for Cognizant-Hack")
    parser.add_argument("--setup-ads", action="store_true", help="Index all sponsor ad campaigns into Pinecone")
    parser.add_argument("--sync-video", type=str, help="Upload scenes and embeddings for a specific Video ID")
    parser.add_argument("--sync-all", action="store_true", help="Sync all processed videos in data/scenes to Pinecone")
    parser.add_argument("--search", type=str, help="Search indexed video scenes using a natural language query")

    args = parser.parse_args()
    mgr = PineconeManager()

    if not mgr.is_available():
        print("⚠️ Pinecone is not currently configured or authenticated.")
        print("   To connect, please add your PINECONE_API_KEY in a .env file:")
        print("   PINECONE_API_KEY=your_key_here")
        return

    if args.setup_ads:
        from recommend_ads_advanced import AdvancedAdRecommender
        recommender = AdvancedAdRecommender()
        mgr.upsert_ad_campaigns(recommender.ad_campaigns)

    elif args.sync_video:
        vid = args.sync_video.strip()
        json_path = os.path.join("data", "scenes", f"{vid}_scenes.json")
        npy_path = os.path.join("data", "scenes", f"{vid}_embeddings.npy")
        if os.path.exists(json_path) and os.path.exists(npy_path):
            with open(json_path, "r", encoding="utf-8") as f:
                scenes_data = json.load(f)
            embeddings = np.load(npy_path)
            mgr.upsert_video_scenes(vid, scenes_data, embeddings)
        else:
            print(f"❌ Scene data files for '{vid}' not found in data/scenes/")

    elif args.sync_all:
        scenes_dir = os.path.join("data", "scenes")
        if not os.path.exists(scenes_dir):
            print("❌ data/scenes directory does not exist.")
            return
        
        synced = 0
        for f in os.listdir(scenes_dir):
            if f.endswith("_scenes.json"):
                vid = f.replace("_scenes.json", "")
                npy_path = os.path.join(scenes_dir, f"{vid}_embeddings.npy")
                if os.path.exists(npy_path):
                    with open(os.path.join(scenes_dir, f), "r", encoding="utf-8") as jf:
                        scenes_data = json.load(jf)
                    embeddings = np.load(npy_path)
                    if mgr.upsert_video_scenes(vid, scenes_data, embeddings):
                        synced += 1
        print(f"\n🎉 Synced {synced} video(s) to Pinecone!")

    elif args.search:
        print(f"\n🔍 Searching indexed video scenes for: \"{args.search}\"...\n")
        results = mgr.search_video_scenes(args.search, top_k=5)
        if not results:
            print("No matching scenes found.")
        else:
            for idx, r in enumerate(results, 1):
                meta = r["metadata"]
                print(f"[{idx}] 🎬 Video: \"{meta.get('video_title')}\" ({meta.get('video_id')})")
                print(f"    ⏱️  Scene #{meta.get('scene_id')} @ {meta.get('start_sec')}s - {meta.get('end_sec')}s")
                print(f"    🖼️  Visuals: \"{meta.get('visual_caption')}\"")
                print(f"    🎭 Mood: {meta.get('emotion_tag')}")
                print(f"    📊 Match Score: {r['score']*100:.1f}%")
                print("-" * 50)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
