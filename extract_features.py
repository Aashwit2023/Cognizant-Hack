import os
import sys
import shutil
import cv2
import json
import re
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

class YouTubeFeatureExtractor:
    def __init__(self, output_dir="data"):
        self.output_dir = output_dir
        self.dirs = {
            "videos": os.path.join(output_dir, "videos"),
            "frames": os.path.join(output_dir, "frames"),
            "transcripts": os.path.join(output_dir, "transcripts"),
            "metadata": os.path.join(output_dir, "metadata"),
            "scenes": os.path.join(output_dir, "scenes"),
        }
        # Create directories if they don't exist
        for path in self.dirs.values():
            os.makedirs(path, exist_ok=True)

    def extract_video_id(self, url):
        """Extracts the video ID from a YouTube URL or direct ID."""
        url = url.strip()
        # Direct 11-char ID
        if re.fullmatch(r"[0-9A-Za-z_-]{11}", url):
            return url
        # Standard URL patterns (watch?v=, youtu.be/, shorts/, embed/)
        match = re.search(r"(?:v=|\/|embed\/|shorts\/)([0-9A-Za-z_-]{11})", url)
        return match.group(1) if match else None

    def is_already_processed(self, video_id):
        """
        Checks if scene analysis has already been completed for this video.
        Returns True if both the scenes JSON and embeddings .npy file exist and are valid.
        """
        json_path = os.path.join(self.dirs["scenes"], f"{video_id}_scenes.json")
        npy_path  = os.path.join(self.dirs["scenes"], f"{video_id}_embeddings.npy")
        if os.path.exists(json_path) and os.path.exists(npy_path):
            # Extra validation: make sure the JSON contains at least 1 scene
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if data.get("total_scenes", 0) > 0:
                    return True
            except Exception:
                pass
        return False

    def cleanup_media(self, video_id, video_path=None):
        """
        Safely deletes heavy temporary media files ONLY after confirming that
        scene analysis outputs (_scenes.json and _embeddings.npy) are valid on disk.
        Keeps: metadata JSON, transcript JSON, scenes JSON, embeddings NPY.
        Removes: raw video MP4, extracted frames folder, scene keyframe image folder.
        """
        json_path = os.path.join(self.dirs["scenes"], f"{video_id}_scenes.json")
        npy_path  = os.path.join(self.dirs["scenes"], f"{video_id}_embeddings.npy")

        # Safety gate: only delete if both output files confirmed to exist and are valid
        if not os.path.exists(json_path) or not os.path.exists(npy_path):
            print("⚠️  Cleanup skipped: scene output files not found. Data may not have been saved correctly.")
            return

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data.get("total_scenes", 0) == 0:
                print("⚠️  Cleanup skipped: scenes JSON has 0 scenes, processing may have failed.")
                return
        except Exception as e:
            print(f"⚠️  Cleanup skipped: could not validate scenes JSON ({e}).")
            return

        print("\n🧹 Scene data confirmed saved. Cleaning up heavy media files...")

        # 1. Delete raw video MP4
        mp4_path = video_path or os.path.join(self.dirs["videos"], f"{video_id}.mp4")
        if os.path.exists(mp4_path):
            os.remove(mp4_path)
            print(f"   🗑️  Deleted video:          {mp4_path}")
        else:
            print(f"   ℹ️  Video not found (already removed): {mp4_path}")

        # 2. Delete extracted frames folder
        frames_dir = os.path.join(self.dirs["frames"], video_id)
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)
            print(f"   🗑️  Deleted frames folder:  {frames_dir}/")

        # 3. Delete scene keyframe images folder
        keyframes_dir = os.path.join(self.dirs["scenes"], video_id)
        if os.path.exists(keyframes_dir):
            shutil.rmtree(keyframes_dir)
            print(f"   🗑️  Deleted keyframes folder: {keyframes_dir}/")

        print("✅ Cleanup complete! Lightweight data retained:")
        print(f"   💾 {json_path}")
        print(f"   💾 {npy_path}")
        print(f"   💾 {os.path.join(self.dirs['metadata'], video_id + '.json')}")
        print(f"   💾 {os.path.join(self.dirs['transcripts'], video_id + '.json')}")

    def download_video_and_metadata(self, url, video_id):
        """Downloads the video file and saves metadata as JSON."""
        print(f"[1/3] Fetching Video and Metadata for {video_id}...")
        
        # If url was just the video ID, format as full YouTube URL for yt_dlp
        if not url.startswith("http://") and not url.startswith("https://"):
            url = f"https://www.youtube.com/watch?v={video_id}"

        video_path = os.path.join(self.dirs["videos"], f"{video_id}.mp4")
        metadata_path = os.path.join(self.dirs["metadata"], f"{video_id}.json")

        # --- YDL OPTS TO BYPASS BOT DETECTION & 403 FORBIDDEN ERROR ---
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'outtmpl': video_path,
            'quiet': False,
            'no_warnings': True,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1',
                'Accept-Language': 'en-US,en;q=0.9',
            },
            'extractor_args': {
                'youtube': {
                    'player_client': ['ios', 'android', 'tv_embedded', 'mweb'],
                }
            }
        }

        # Auto-detect cookies.txt if available
        cookie_paths = ["cookies.txt", os.path.join(self.output_dir, "cookies.txt")]
        for cp in cookie_paths:
            if os.path.exists(cp):
                print(f"🍪 Using cookies from {cp}")
                ydl_opts['cookiefile'] = cp
                break

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            
            metadata = {
                "id": info.get("id", video_id),
                "title": info.get("title", ""),
                "description": info.get("description", ""),
                "tags": info.get("tags", []),
                "categories": info.get("categories", []),
                "view_count": info.get("view_count", 0),
                "like_count": info.get("like_count", 0),
                "duration": info.get("duration", 0),
                "channel": info.get("uploader", "")
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
                
        print(f"✅ Video saved to {video_path}")
        print(f"✅ Metadata saved to {metadata_path}")
        return video_path

    def extract_transcript(self, video_id):
        """Fetches the transcript with timestamps safely."""
        print(f"[2/3] Fetching Transcript for {video_id}...")
        transcript_path = os.path.join(self.dirs["transcripts"], f"{video_id}.json")
        
        try:
            # Support youtube-transcript-api v1.x and legacy versions
            transcript_data = []
            try:
                ytt = YouTubeTranscriptApi()
                try:
                    transcript_obj = ytt.fetch(video_id, languages=['en', 'hi'])
                except Exception:
                    # Fallback: list transcripts and pick the first available
                    transcript_list = ytt.list(video_id)
                    try:
                        transcript_obj = transcript_list.find_transcript(['en', 'hi'])
                    except Exception:
                        transcript_obj = next(iter(transcript_list))
                    transcript_obj = transcript_obj.fetch()

                if hasattr(transcript_obj, 'to_raw_data'):
                    transcript_data = transcript_obj.to_raw_data()
                elif hasattr(transcript_obj, 'to_dict'):
                    transcript_data = transcript_obj.to_dict()
                else:
                    transcript_data = [
                        {"text": getattr(item, 'text', str(item)), "start": getattr(item, 'start', 0.0), "duration": getattr(item, 'duration', 0.0)}
                        if hasattr(item, 'text') else item
                        for item in transcript_obj
                    ]
            except Exception:
                # Legacy API fallback
                try:
                    transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'hi'])
                except Exception:
                    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                    transcript_obj = transcript_list.find_transcript(['en', 'hi'])
                    transcript_data = transcript_obj.fetch()

            with open(transcript_path, 'w', encoding='utf-8') as f:
                json.dump(transcript_data, f, indent=4, ensure_ascii=False)
            print(f"✅ Transcript saved to {transcript_path}")
            
        except Exception as e:
            print(f"❌ Could not fetch transcript: {e}")
            print("Note: The video might not have any closed captions available.")
            # Write empty list to avoid missing file errors in downstream pipeline
            with open(transcript_path, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=4, ensure_ascii=False)

    def extract_frames(self, video_path, video_id, interval_seconds=5):
        """Extracts 1 frame every X seconds to save storage and processing time."""
        print(f"[3/3] Extracting frames (1 every {interval_seconds} seconds)...")
        
        frame_dir = os.path.join(self.dirs["frames"], video_id)
        os.makedirs(frame_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        fps = round(cap.get(cv2.CAP_PROP_FPS) or 0)
        
        # Prevent division by zero if video fails to load
        if fps <= 0:
            print("❌ Could not read video frames.")
            cap.release()
            return
            
        frame_interval = max(int(fps * interval_seconds), 1)
        count = 0
        saved_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if count % frame_interval == 0:
                timestamp = count // fps 
                frame_filename = os.path.join(frame_dir, f"frame_{timestamp}s.jpg")
                cv2.imwrite(frame_filename, frame)
                saved_count += 1
                
            count += 1
            
        cap.release()
        print(f"✅ Extracted {saved_count} frames to {frame_dir}/")

    def process_url(self, url, run_scene_analysis=True, auto_cleanup=True, run_ad_recommendations=True):
        """
        Main pipeline function.
        - auto_cleanup: If True, deletes video, frames, and keyframe images AFTER
          scene analysis is confirmed complete and data is saved.
        - run_ad_recommendations: If True, runs hybrid ad recommendation right after
          processing completes.
        """
        video_id = self.extract_video_id(url)
        if not video_id:
            print("❌ Invalid YouTube URL or Video ID")
            return

        # ── SMART CACHE: Skip full pipeline if already processed ──────────────
        if self.is_already_processed(video_id):
            print(f"\n✅ Video '{video_id}' has already been processed.")
            print(f"   Scene data and embeddings found in data/scenes/.")
            print(f"   Skipping download & re-processing.\n")
            if run_ad_recommendations:
                self._run_recommendations(video_id)
            return

        print(f"Starting extraction for Video ID: {video_id}\n" + "-"*40)
        
        # 1. Video & Metadata
        try:
            video_path = self.download_video_and_metadata(url, video_id)
        except Exception as e:
            print(f"❌ Failed to download video: {e}")
            return
        
        # 2. Transcript
        self.extract_transcript(video_id)
        
        # 3. Frames (Extracting 1 frame every 5 seconds)
        self.extract_frames(video_path, video_id, interval_seconds=5)
        
        print("-" * 40 + "\n🎉 Base Extraction Complete!")

        # 4. Multi-Modal Scene Analysis Pipeline (Steps 1 to 5)
        scene_analysis_ok = False
        if run_scene_analysis:
            try:
                from scene_analyzer import SceneAnalyzer
                analyzer = SceneAnalyzer(output_dir=self.output_dir)
                analyzer.analyze_video(video_path, video_id)
                scene_analysis_ok = True
            except Exception as e:
                print(f"⚠️ Scene analysis failed or skipped: {e}")

        # 5. Auto Cleanup: Delete video, frames & keyframes ONLY after confirmed save
        if auto_cleanup and scene_analysis_ok:
            self.cleanup_media(video_id, video_path=video_path)
        elif auto_cleanup and not scene_analysis_ok:
            print("\n⚠️  Cleanup skipped because scene analysis did not complete successfully.")
            print(f"   The video file is preserved at: {video_path}")

        # 6. Run Ad Recommendations using lightweight saved data
        if run_ad_recommendations and scene_analysis_ok:
            self._run_recommendations(video_id)

    def _run_recommendations(self, video_id):
        """Runs the Hybrid Ad Recommender on the processed video."""
        print("\n" + "="*60)
        print("🎯 Running Hybrid Ad Recommendation Engine...")
        print("="*60)
        try:
            from recommend_ads_advanced import AdvancedAdRecommender
            recommender = AdvancedAdRecommender(data_dir=self.output_dir)
            recommender.recommend_for_video(video_id)
        except Exception as e:
            print(f"⚠️ Ad recommendation failed: {e}")
            print(f"   You can run it manually: python recommend_ads_advanced.py {video_id}")


# --- Run the Script ---
if __name__ == "__main__":
    if len(sys.argv) > 1:
        youtube_url = sys.argv[1]
    else:
        youtube_url = input("Please enter the YouTube Video URL or Video ID: ")
    
    extractor = YouTubeFeatureExtractor()
    extractor.process_url(youtube_url)