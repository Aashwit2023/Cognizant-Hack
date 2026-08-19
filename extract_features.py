import os
import cv2
import json
import re
import yt_dlp
# EXACT IMPORT NEEDED FOR THE TRANSCRIPT API
from youtube_transcript_api import YouTubeTranscriptApi

class YouTubeFeatureExtractor:
    def __init__(self, output_dir="data"):
        self.output_dir = output_dir
        self.dirs = {
            "videos": os.path.join(output_dir, "videos"),
            "frames": os.path.join(output_dir, "frames"),
            "transcripts": os.path.join(output_dir, "transcripts"),
            "metadata": os.path.join(output_dir, "metadata")
        }
        # Create directories if they don't exist
        for path in self.dirs.values():
            os.makedirs(path, exist_ok=True)

    def extract_video_id(self, url):
        """Extracts the video ID from a YouTube URL."""
        match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
        return match.group(1) if match else None

    def download_video_and_metadata(self, url, video_id):
        """Downloads the video file and saves metadata as JSON."""
        print(f"[1/3] Fetching Video and Metadata for {video_id}...")
        
        video_path = os.path.join(self.dirs["videos"], f"{video_id}.mp4")
        metadata_path = os.path.join(self.dirs["metadata"], f"{video_id}.json")

        # --- YDL OPTS TO BYPASS 403 FORBIDDEN ERROR ---
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': video_path,
            'quiet': True,
            'no_warnings': True,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept-Language': 'en-US,en;q=0.9',
            },
            'extractor_args': {
                'youtube': {
                    'player_client': ['android', 'web'],
                }
            }
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info
            info = ydl.extract_info(url, download=True)
            
            # Save relevant metadata
            metadata = {
                "id": info.get("id"),
                "title": info.get("title"),
                "description": info.get("description"),
                "tags": info.get("tags", []),
                "categories": info.get("categories", []),
                "view_count": info.get("view_count"),
                "like_count": info.get("like_count"),
                "duration": info.get("duration"),
                "channel": info.get("uploader")
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
                
        print(f"✅ Video saved to {video_path}")
        print(f"✅ Metadata saved to {metadata_path}")
        return video_path

    # --- UPDATED TRANSCRIPT FUNCTION ---
    def extract_transcript(self, video_id):
        """Fetches the transcript with timestamps safely."""
        print(f"[2/3] Fetching Transcript for {video_id}...")
        transcript_path = os.path.join(self.dirs["transcripts"], f"{video_id}.json")
        
        try:
            # Support both youtube-transcript-api v1.x (instance) and legacy (static)
            try:
                ytt = YouTubeTranscriptApi()
                transcript_list = ytt.list(video_id)
            except (AttributeError, TypeError):
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try getting English or Hindi first
            try:
                transcript_obj = transcript_list.find_transcript(['en', 'hi'])
            except Exception:
                # Fallback: take the first available transcript
                transcript_obj = next(iter(transcript_list))
                
            fetched = transcript_obj.fetch()
            
            # Helper to convert transcript objects into serializable JSON format
            def serialize(obj):
                if hasattr(obj, 'to_raw_data'):
                    return obj.to_raw_data()
                if hasattr(obj, 'to_dict'):
                    return obj.to_dict()
                if isinstance(obj, list):
                    return [serialize(item) for item in obj]
                if isinstance(obj, dict):
                    return {k: serialize(v) for k, v in obj.items()}
                if hasattr(obj, '__dict__'):
                    return {k: serialize(v) for k, v in obj.__dict__.items()}
                return obj

            transcript_data = serialize(fetched)
                    
            # Save to JSON
            with open(transcript_path, 'w', encoding='utf-8') as f:
                json.dump(transcript_data, f, indent=4, ensure_ascii=False)
            print(f"✅ Transcript saved to {transcript_path}")
            
        except Exception as e:
            print(f"❌ Could not fetch transcript: {e}")
            print("Note: The video might not have any closed captions available.")

    def extract_frames(self, video_path, video_id, interval_seconds=5):
        """Extracts 1 frame every X seconds to save storage and processing time."""
        print(f"[3/3] Extracting frames (1 every {interval_seconds} seconds)...")
        
        frame_dir = os.path.join(self.dirs["frames"], video_id)
        os.makedirs(frame_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        
        # Prevent division by zero if video fails to load
        if fps == 0:
            print("❌ Could not read video frames.")
            return
            
        frame_interval = fps * interval_seconds
        
        count = 0
        saved_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if count % frame_interval == 0:
                # Time in seconds
                timestamp = count // fps 
                frame_filename = os.path.join(frame_dir, f"frame_{timestamp}s.jpg")
                cv2.imwrite(frame_filename, frame)
                saved_count += 1
                
            count += 1
            
        cap.release()
        print(f"✅ Extracted {saved_count} frames to {frame_dir}/")

    def process_url(self, url, run_scene_analysis=True):
        """Main pipeline function."""
        video_id = self.extract_video_id(url)
        if not video_id:
            print("❌ Invalid YouTube URL")
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
        if run_scene_analysis:
            try:
                from scene_analyzer import SceneAnalyzer
                analyzer = SceneAnalyzer(output_dir=self.output_dir)
                analyzer.analyze_video(video_path, video_id)
            except Exception as e:
                print(f"⚠️ Scene analysis failed or skipped: {e}")


# --- Run the Script ---
if __name__ == "__main__":
    youtube_url = input("Please enter the YouTube Video URL: ")
    
    extractor = YouTubeFeatureExtractor()
    extractor.process_url(youtube_url)