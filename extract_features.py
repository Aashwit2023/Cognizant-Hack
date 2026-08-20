import os
import sys
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
            "metadata": os.path.join(output_dir, "metadata")
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

    def process_url(self, url, run_scene_analysis=True):
        """Main pipeline function."""
        video_id = self.extract_video_id(url)
        if not video_id:
            print("❌ Invalid YouTube URL or Video ID")
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
    if len(sys.argv) > 1:
        youtube_url = sys.argv[1]
    else:
        youtube_url = input("Please enter the YouTube Video URL or Video ID: ")
    
    extractor = YouTubeFeatureExtractor()
    extractor.process_url(youtube_url)