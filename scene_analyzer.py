import os
import json
import time
import cv2
import numpy as np
from PIL import Image
import torch

class SceneAnalyzer:
    # Class-level model caches to ensure each model is loaded ONLY ONCE across the entire application runtime
    _cached_caption_processor = None
    _cached_caption_model = None
    _cached_emotion_classifier = None
    _cached_embedding_model = None
    _model_load_logged = {"blip": False, "emotion": False, "embedding": False}

    def __init__(self, output_dir="data", device=None):
        self.output_dir = output_dir
        self.scenes_dir = os.path.join(output_dir, "scenes")
        os.makedirs(self.scenes_dir, exist_ok=True)
        
        # 1. Device check
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        if self.device == "cuda":
            device_name = torch.cuda.get_device_name(0)
            print(f"⚙️ SceneAnalyzer Device: CUDA (GPU: {device_name})")
        else:
            # CPU-specific optimization: Maximize multi-threaded PyTorch CPU performance
            cpu_threads = min(os.cpu_count() or 4, 8)
            torch.set_num_threads(cpu_threads)
            print(f"⚙️ SceneAnalyzer Device: CPU (Optimized with {cpu_threads} CPU worker threads)")

    # ==========================================
    # MODEL LOADERS (LOADED ONCE & CACHED)
    # ==========================================
    def _load_caption_model(self):
        if SceneAnalyzer._cached_caption_model is None:
            t0 = time.time()
            from transformers import BlipProcessor, BlipForConditionalGeneration
            model_id = "Salesforce/blip-image-captioning-base"
            
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            SceneAnalyzer._cached_caption_processor = BlipProcessor.from_pretrained(model_id)
            SceneAnalyzer._cached_caption_model = BlipForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=dtype
            ).to(self.device)
            SceneAnalyzer._cached_caption_model.eval()
            
            if not SceneAnalyzer._model_load_logged["blip"]:
                print(f"✅ [Model Load] BLIP model loaded once on {self.device.upper()} (took {time.time() - t0:.2f}s)")
                SceneAnalyzer._model_load_logged["blip"] = True

    def _load_emotion_classifier(self):
        if SceneAnalyzer._cached_emotion_classifier is None:
            t0 = time.time()
            from transformers import pipeline
            device_idx = 0 if self.device == "cuda" else -1
            SceneAnalyzer._cached_emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=device_idx,
                top_k=1
            )
            if not SceneAnalyzer._model_load_logged["emotion"]:
                print(f"✅ [Model Load] Emotion classifier pipeline loaded once on {self.device.upper()} (took {time.time() - t0:.2f}s)")
                SceneAnalyzer._model_load_logged["emotion"] = True

    def _load_embedding_model(self):
        if SceneAnalyzer._cached_embedding_model is None:
            t0 = time.time()
            from sentence_transformers import SentenceTransformer
            SceneAnalyzer._cached_embedding_model = SentenceTransformer(
                "paraphrase-multilingual-MiniLM-L12-v2", # YAHAN CHANGE KIYA HAI
                device=self.device
            )
            if not SceneAnalyzer._model_load_logged["embedding"]:
                print(f"✅ [Model Load] SentenceTransformer loaded once on {self.device.upper()} (took {time.time() - t0:.2f}s)")
                SceneAnalyzer._model_load_logged["embedding"] = True

    # ==========================================
    # 1. GROUP FRAMES INTO SCENES
    # ==========================================
    def detect_scenes(self, video_path, video_id, threshold=30.0, min_scene_sec=3.0, fallback_interval_sec=10.0):
        """
        Detects meaningful scene boundaries using PySceneDetect (with minimum duration filter)
        and extracts exactly 1 midpoint keyframe per scene.
        """
        t0 = time.time()
        print(f"\n[Step 1/5] Detecting scenes for {video_id}...")
        scene_list = []
        
        video_keyframes_dir = os.path.join(self.scenes_dir, video_id)
        os.makedirs(video_keyframes_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        min_scene_frames = int(fps * min_scene_sec)
        cap.release()

        try:
            from scenedetect import detect, ContentDetector
            # min_scene_len prevents micro-scenes (rapid cuts), saving huge amounts of CPU processing time
            detected_scenes = detect(video_path, ContentDetector(threshold=threshold, min_scene_len=min_scene_frames))
            
            if detected_scenes and len(detected_scenes) > 1:
                for idx, (start_time, end_time) in enumerate(detected_scenes):
                    scene_list.append({
                        "scene_id": idx + 1,
                        "start_sec": round(start_time.get_seconds(), 2),
                        "end_sec": round(end_time.get_seconds(), 2),
                        "duration_sec": round(end_time.get_seconds() - start_time.get_seconds(), 2)
                    })
                print(f"🎬 PySceneDetect identified {len(scene_list)} distinct scenes (>= {min_scene_sec}s each).")
        except Exception as e:
            print(f"⚠️ PySceneDetect fallback triggered ({e}).")

        # Fallback: Fixed-interval segmentation if no cuts found
        if len(scene_list) <= 1:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            total_duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            scene_list = []
            curr_start = 0.0
            scene_idx = 1
            while curr_start < total_duration:
                curr_end = min(curr_start + fallback_interval_sec, total_duration)
                scene_list.append({
                    "scene_id": scene_idx,
                    "start_sec": round(curr_start, 2),
                    "end_sec": round(curr_end, 2),
                    "duration_sec": round(curr_end - curr_start, 2)
                })
                curr_start = curr_end
                scene_idx += 1
            print(f"🎬 Created {len(scene_list)} interval-based scenes ({fallback_interval_sec}s intervals).")

        # Extract exactly 1 representative midpoint keyframe per scene
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        for scene in scene_list:
            midpoint_sec = (scene["start_sec"] + scene["end_sec"]) / 2.0
            frame_idx = int(midpoint_sec * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            frame_filename = f"scene_{scene['scene_id']:03d}_{round(midpoint_sec, 1)}s.jpg"
            frame_path = os.path.join(video_keyframes_dir, frame_filename)
            
            if ret:
                cv2.imwrite(frame_path, frame)
                scene["keyframe_path"] = frame_path
                scene["keyframe_time_sec"] = round(midpoint_sec, 2)
            else:
                scene["keyframe_path"] = None
                scene["keyframe_time_sec"] = None

        cap.release()
        elapsed = time.time() - t0
        print(f"✅ Scene Detection complete: {len(scene_list)} scenes extracted ({elapsed:.2f}s)")
        return scene_list, elapsed

    # ==========================================
    # 2. CAPTION EACH FRAME (OPTIMIZED BLIP FOR CPU)
    # ==========================================
    def caption_scenes(self, scenes, batch_size=4):
        """
        Batched and resized BLIP captioning with torch.no_grad() and greedy search for CPU.
        """
        t0 = time.time()
        print(f"\n[Step 2/5] Generating visual captions for {len(scenes)} scenes (batch_size={batch_size})...")
        self._load_caption_model()
        
        processor = SceneAnalyzer._cached_caption_processor
        model = SceneAnalyzer._cached_caption_model
        resampling = getattr(Image, 'Resampling', Image).LANCZOS
        
        for i in range(0, len(scenes), batch_size):
            batch_scenes = scenes[i:i + batch_size]
            batch_images = []
            valid_indices = []
            
            for local_idx, scene in enumerate(batch_scenes):
                fpath = scene.get("keyframe_path")
                if fpath and os.path.exists(fpath):
                    try:
                        img = Image.open(fpath).convert('RGB')
                        # Downscale/resize to 384x384 for fast CPU processing
                        img = img.resize((384, 384), resampling)
                        batch_images.append(img)
                        valid_indices.append(local_idx)
                    except Exception:
                        scene["visual_caption"] = ""
                else:
                    scene["visual_caption"] = ""
                    
            if batch_images:
                try:
                    inputs = processor(batch_images, return_tensors="pt").to(self.device)
                    if self.device == "cuda":
                        inputs = {k: v.to(torch.float16) if v.dtype == torch.float32 else v for k, v in inputs.items()}
                    
                    # Greedy search (num_beams=1, max_new_tokens=20) is 3x-4x faster on CPU
                    with torch.no_grad():
                        out = model.generate(**inputs, max_new_tokens=20, num_beams=1)
                        captions = processor.batch_decode(out, skip_special_tokens=True)
                        
                    for local_idx, cap_text in zip(valid_indices, captions):
                        batch_scenes[local_idx]["visual_caption"] = cap_text.strip()
                except Exception as e:
                    print(f"⚠️ Batch captioning fallback: {e}")
                    for local_idx in valid_indices:
                        batch_scenes[local_idx]["visual_caption"] = ""

        elapsed = time.time() - t0
        avg_ms = (elapsed / max(len(scenes), 1)) * 1000
        print(f"✅ BLIP Captioning complete: {len(scenes)} captions generated ({elapsed:.2f}s, ~{avg_ms:.1f}ms/scene)")
        return scenes, elapsed

    # ==========================================
    # 3. MATCH TRANSCRIPT TO SCENES
    # ==========================================
    def match_transcript_to_scenes(self, scenes, transcript_path):
        """
        Slices transcript timestamps into scene buckets.
        """
        t0 = time.time()
        print(f"\n[Step 3/5] Matching transcript dialogue to scenes...")
        
        transcript_data = []
        if os.path.exists(transcript_path):
            try:
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    transcript_data = json.load(f)
            except Exception as e:
                print(f"⚠️ Failed to read transcript: {e}")

        for scene in scenes:
            scene_start = scene["start_sec"]
            scene_end = scene["end_sec"]
            
            matched = []
            for item in transcript_data:
                t_start = item.get("start", 0.0)
                t_duration = item.get("duration", 0.0)
                t_end = t_start + t_duration
                
                if t_end >= scene_start and t_start <= scene_end:
                    text = item.get("text", "").strip()
                    if text and text != "[Music]":
                        matched.append(text)
            
            scene["transcript_dialogue"] = " ".join(matched).strip()
            
        elapsed = time.time() - t0
        print(f"✅ Transcript alignment complete ({elapsed:.2f}s)")
        return scenes, elapsed

    # ==========================================
    # 4. GET EMOTION / TONE (BATCHED)
    # ==========================================
    def classify_scene_emotions(self, scenes, batch_size=32):
        """
        Batched emotion classification across all scene texts at once.
        """
        t0 = time.time()
        print(f"\n[Step 4/5] Classifying emotional tone for {len(scenes)} scenes in batches...")
        self._load_emotion_classifier()
        classifier = SceneAnalyzer._cached_emotion_classifier

        texts_to_classify = []
        for scene in scenes:
            dialogue = scene.get("transcript_dialogue", "").strip()
            caption = scene.get("visual_caption", "").strip()
            
            if dialogue:
                text = dialogue[:512]
            elif caption:
                text = f"A scene depicting {caption}"
            else:
                text = "neutral scene"
            texts_to_classify.append(text)

        try:
            with torch.no_grad():
                results = classifier(texts_to_classify, batch_size=batch_size, truncation=True)
                
            for idx, res in enumerate(results):
                top_pred = res[0] if isinstance(res, list) else res
                scenes[idx]["emotion_tag"] = top_pred.get("label", "neutral")
                scenes[idx]["emotion_confidence"] = round(float(top_pred.get("score", 0.0)), 3)
        except Exception as e:
            print(f"⚠️ Emotion classification error: {e}")
            for scene in scenes:
                scene["emotion_tag"] = "neutral"
                scene["emotion_confidence"] = 0.0

        elapsed = time.time() - t0
        print(f"✅ Emotion tagging complete ({elapsed:.2f}s)")
        return scenes, elapsed

    # ==========================================
    # 5. COMBINE INTO PROFILES & EMBEDDINGS (BATCHED)
    # ==========================================
    def build_scene_profiles_and_embeddings(self, scenes, video_id, metadata=None, batch_size=64):
        """
        Batched sentence embedding generation with torch.no_grad().
        """
        t0 = time.time()
        print(f"\n[Step 5/5] Building profiles & generating {len(scenes)} embeddings...")
        self._load_embedding_model()
        model = SceneAnalyzer._cached_embedding_model

        profile_texts = []
        for scene in scenes:
            caption = scene.get("visual_caption", "").strip()
            dialogue = scene.get("transcript_dialogue", "").strip()
            emotion = scene.get("emotion_tag", "neutral")
            
            dialogue_str = f' Dialogue: "{dialogue}".' if dialogue else ""
            profile_text = f"Visuals: {caption}.{dialogue_str} Tone: {emotion}."
            scene["profile_text"] = profile_text
            profile_texts.append(profile_text)

        with torch.no_grad():
            embeddings = model.encode(
                profile_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True
            )
        
        for idx, scene in enumerate(scenes):
            scene["embedding"] = embeddings[idx].tolist()

        # Save Scene Data JSON
        output_json_path = os.path.join(self.scenes_dir, f"{video_id}_scenes.json")
        output_data = {
            "video_id": video_id,
            "metadata": metadata or {},
            "total_scenes": len(scenes),
            "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2", # YAHAN CHANGE KIYA HAI
            "embedding_dim": int(embeddings.shape[1]) if len(embeddings) > 0 else 384,
            "scenes": scenes
        }
        
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
            
        # Save NumPy Embeddings (.npy)
        output_npy_path = os.path.join(self.scenes_dir, f"{video_id}_embeddings.npy")
        np.save(output_npy_path, embeddings)

        elapsed = time.time() - t0
        print(f"✅ Saved scene profiles to {output_json_path}")
        print(f"✅ Saved embeddings matrix {embeddings.shape} to {output_npy_path} ({elapsed:.2f}s)")
        return output_data, elapsed

    # ==========================================
    # FULL END-TO-END PIPELINE WITH TIMING SUMMARY
    # ==========================================
    def analyze_video(self, video_path, video_id, transcript_path=None, metadata_path=None):
        total_start = time.time()
        print(f"\n" + "="*60)
        print(f"🚀 RUNNING OPTIMIZED SCENE PIPELINE FOR: {video_id}")
        print("="*60)

        if transcript_path is None:
            transcript_path = os.path.join(self.output_dir, "transcripts", f"{video_id}.json")
        if metadata_path is None:
            metadata_path = os.path.join(self.output_dir, "metadata", f"{video_id}.json")

        metadata = {}
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except Exception:
                pass

        # Step 1: Detect Scenes
        scenes, t_step1 = self.detect_scenes(video_path, video_id)
        
        # Step 2: BLIP Frame Captions
        scenes, t_step2 = self.caption_scenes(scenes)
        
        # Step 3: Match Transcript
        scenes, t_step3 = self.match_transcript_to_scenes(scenes, transcript_path)
        
        # Step 4: Classify Emotion/Tone
        scenes, t_step4 = self.classify_scene_emotions(scenes)
        
        # Step 5: Profile Fusion & Sentence Embeddings
        result, t_step5 = self.build_scene_profiles_and_embeddings(scenes, video_id, metadata)
        
        total_time = time.time() - total_start

        # Print Timing Breakdown
        print("\n" + "="*60)
        print("⏱️  SCENE PIPELINE TIMING BREAKDOWN")
        print("="*60)
        print(f"1. Scene Detection (PySceneDetect):   {t_step1:6.2f}s  ({(t_step1/total_time)*100:4.1f}%)")
        print(f"2. BLIP Keyframe Captioning:         {t_step2:6.2f}s  ({(t_step2/total_time)*100:4.1f}%)")
        print(f"3. Transcript Alignment:             {t_step3:6.2f}s  ({(t_step3/total_time)*100:4.1f}%)")
        print(f"4. Emotion Classification (Batched): {t_step4:6.2f}s  ({(t_step4/total_time)*100:4.1f}%)")
        print(f"5. Sentence Embeddings (Batched):    {t_step5:6.2f}s  ({(t_step5/total_time)*100:4.1f}%)")
        print("-" * 60)
        print(f"🏆 Total Processing Time:            {total_time:6.2f}s ({len(scenes)} scenes)")
        print("="*60 + "\n")

        return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        vid_id = sys.argv[1]
    else:
        vid_id = input("Enter Video ID to analyze (e.g., xLTCivIB4kU): ").strip()
        
    vid_path = os.path.join("data", "videos", f"{vid_id}.mp4")
    if not os.path.exists(vid_path):
        print(f"❌ Video not found at {vid_path}")
    else:
        analyzer = SceneAnalyzer()
        analyzer.analyze_video(vid_path, vid_id)
