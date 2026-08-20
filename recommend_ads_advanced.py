import os
import sys
import json
import re
import math
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

def clean_video_id(input_str):
    """Extracts a clean video ID from either a URL or raw ID."""
    input_str = input_str.strip()
    if re.fullmatch(r"[0-9A-Za-z_-]{11}", input_str):
        return input_str
    match = re.search(r"(?:v=|\/|embed\/|shorts\/)([0-9A-Za-z_-]{11})", input_str)
    return match.group(1) if match else input_str


# =====================================================================
# 1. LIGHTWEIGHT BM25 LEXICAL RETRIEVAL ENGINE
# =====================================================================
class BM25Retriever:
    """
    Lightweight BM25 lexical engine supporting multilingual (English + transliterated Hindi)
    token matching with field-aware weighting.
    """
    STOPWORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", 
        "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", 
        "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", 
        "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", 
        "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", 
        "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", 
        "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", 
        "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", 
        "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", 
        "why", "with", "would", "you", "your", "yours", "yourself", "yourselves",
        # Common Hindi stopwords in Devanagari & Latin transliteration
        "hai", "hain", "ke", "ki", "ka", "ko", "se", "me", "mein", "par", "ye", "yeh", "wo", "woh", 
        "to", "bhi", "tha", "thi", "the", "kar", "karna", "raha", "rahe", "rahi", "kya", "aur",
        "है", "हैं", "के", "की", "का", "को", "से", "में", "पर", "ये", "यह", "वो", "वह", "तो", "भी", "था", "थी", "थे", "कर", "करना", "रहा", "रहे", "रही", "क्या", "और"
    }

    def __init__(self, corpus_docs, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus_docs)
        self.doc_lengths = []
        self.doc_term_freqs = []
        self.idf = {}
        
        # Tokenize and index corpus documents
        doc_freqs = Counter()
        for doc in corpus_docs:
            tokens = self.tokenize(doc)
            self.doc_lengths.append(len(tokens))
            tf = Counter(tokens)
            self.doc_term_freqs.append(tf)
            for token in tf.keys():
                doc_freqs[token] += 1
                
        self.avgdl = (sum(self.doc_lengths) / self.corpus_size) if self.corpus_size > 0 else 1.0
        
        # Precompute Robertson-Spärck Jones IDF
        for token, df in doc_freqs.items():
            self.idf[token] = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))

    @staticmethod
    def tokenize(text):
        if not text:
            return []
        text = text.lower()
        # Extract unicode word tokens (supports English, Hindi Devanagari, digits)
        tokens = re.findall(r"[\w']+", text)
        return [t for t in tokens if len(t) > 1 and t not in BM25Retriever.STOPWORDS]

    def get_scores(self, query_text):
        query_tokens = self.tokenize(query_text)
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        if not query_tokens or self.corpus_size == 0:
            return scores

        for token in query_tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]
            for doc_idx, tf_dict in enumerate(self.doc_term_freqs):
                freq = tf_dict.get(token, 0)
                if freq == 0:
                    continue
                doc_len = self.doc_lengths[doc_idx]
                numerator = freq * (self.k1 + 1.0)
                denominator = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                scores[doc_idx] += idf_val * (numerator / denominator)

        # Normalize BM25 scores to [0, 1] range
        max_score = np.max(scores) if len(scores) > 0 else 0.0
        if max_score > 0:
            scores = scores / max_score
        return scores


# =====================================================================
# 2. SCENE CONTEXT & ENVIRONMENT CLASSIFIER
# =====================================================================
class SceneContextClassifier:
    """
    Detects scene environment, visual setting, activity, and atmosphere
    from multimodal scene profiles (visual caption, dialogue, emotion).
    """
    ENVIRONMENT_SIGNALS = {
        "tech_workspace": {
            "keywords": ["laptop", "computer", "desk", "monitor", "screen", "keyboard", "coding", "software", "programming", "office", "tech", "gadget", "processor", "ram", "display", "sitting on couch with laptop", "sitting at desk"],
            "boost_categories": ["Technology", "Laptops", "Smartphones", "Tech Upskilling", "Gaming Gear"]
        },
        "kitchen_dining": {
            "keywords": ["kitchen", "cooking", "food", "table", "eating", "dish", "plate", "meal", "recipe", "snack", "hungry", "restaurant", "lunch", "dinner", "breakfast", "cutting", "frying"],
            "boost_categories": ["Food & Grocery", "Snacks"]
        },
        "gym_fitness": {
            "keywords": ["gym", "workout", "fitness", "running", "exercise", "sports", "weights", "athlete", "training", "jogging", "sweat", "muscle", "marathon"],
            "boost_categories": ["Fitness & Wellness", "Sports Gear"]
        },
        "outdoor_travel": {
            "keywords": ["mountain", "beach", "hotel", "travel", "vacation", "trip", "flight", "resort", "landscape", "nature", "outdoor", "road", "scenic", "city street", "walking outside"],
            "boost_categories": ["Travel & Holidays", "Automobile"]
        },
        "gaming_entertainment": {
            "keywords": ["gaming", "game", "controller", "console", "playstation", "xbox", "esports", "headset", "rgb", "gameplay", "fps", "streamer"],
            "boost_categories": ["Gaming Setup", "Consoles", "Fantasy Sports"]
        },
        "classroom_study": {
            "keywords": ["book", "whiteboard", "classroom", "student", "study", "exam", "reading", "teacher", "lecture", "paper", "pen", "notes"],
            "boost_categories": ["Education & Coaching", "Tech Upskilling", "Language Learning"]
        },
        "automobile_transit": {
            "keywords": ["car", "vehicle", "driving", "bike", "motorcycle", "road", "traffic", "steering wheel", "suv", "helmet", "cruiser"],
            "boost_categories": ["Automobile", "Travel & Holidays"]
        }
    }

    @classmethod
    def compute_scene_affinity(cls, scene_visuals, scene_dialogue, scene_emotion, campaign_profile):
        """
        Calculates affinity score (0.0 - 1.0) based on scene environment, activity, and emotional tone.
        """
        combined_text = f"{scene_visuals} {scene_dialogue}".lower()
        score = 0.0
        
        # 1. Target Visual / Environment Match
        target_envs = campaign_profile.get("target_environments", [])
        for env in target_envs:
            env_data = cls.ENVIRONMENT_SIGNALS.get(env, {})
            keywords = env_data.get("keywords", [])
            matches = sum(1 for kw in keywords if kw in combined_text)
            if matches > 0:
                score += min(0.4, 0.15 * matches)

        # 2. Direct Visual Keyword Match from BLIP Caption
        target_visual_cues = campaign_profile.get("target_visual_cues", [])
        visual_caption_lower = scene_visuals.lower()
        for cue in target_visual_cues:
            if cue.lower() in visual_caption_lower:
                score += 0.25

        # 3. Emotion / Atmosphere Alignment
        preferred_emotions = campaign_profile.get("preferred_emotions", ["neutral", "joy", "surprise"])
        if scene_emotion.lower() in preferred_emotions:
            score += 0.15
        elif scene_emotion.lower() in ["sadness", "anger", "fear"]:
            # Suppress commercial ad placement in negative / sorrowful scene context
            score -= 0.25

        # 4. Activity Alignment
        target_activities = campaign_profile.get("target_activities", [])
        for act in target_activities:
            if act.lower() in combined_text:
                score += 0.2

        return float(np.clip(score, 0.0, 1.0))


# =====================================================================
# 3. ADVANCED HYBRID AD RECOMMENDER
# =====================================================================
class AdvancedAdRecommender:
    def __init__(self, data_dir="data"):
        self.scenes_dir = os.path.join(data_dir, "scenes")
        
        print("🤖 Initializing Multi-Modal Hybrid Ad Recommender...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ad_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=self.device)
        
        # --- STRUCTURED AD CAMPAIGN DATABASE (Scene-Centric Attributes) ---
        self.ad_campaigns = {
            # 💻 Technology, Gadgets & Laptops
            "Apple MacBook Pro & iPad": {
                "category": "Technology",
                "brand": "Apple",
                "semantic_profile": "Professional workspace with laptop, software development, coding on screen, high performance creative video editing, macOS productivity, retina display computer desk setup.",
                "keywords": "apple macbook pro ipad laptop macos m3 chip coding programming video editing retina display computer desk workstation rendering",
                "target_environments": ["tech_workspace", "classroom_study"],
                "target_visual_cues": ["laptop", "computer", "desk", "screen", "couch with laptop", "sitting at desk"],
                "target_activities": ["coding", "editing", "working", "studying"],
                "preferred_emotions": ["neutral", "joy", "surprise"]
            },
            "Budget & Gaming Laptops (HP/Lenovo/Asus)": {
                "category": "Technology",
                "brand": "HP / Lenovo / Asus",
                "semantic_profile": "Student desk setup, best budget laptop for study and coding, buying new laptop under 50000, unboxing computer hardware, intel core i5 ryzen benchmark display specs gaming.",
                "keywords": "laptop under 50000 student laptop budget computer intel i5 ryzen 16gb ram ssd benchmark display specs review gaming naya laptop kharidna",
                "target_environments": ["tech_workspace", "classroom_study", "gaming_entertainment"],
                "target_visual_cues": ["laptop", "computer", "sitting on a couch with a laptop", "desk"],
                "target_activities": ["reviewing", "studying", "gaming", "unboxing"],
                "preferred_emotions": ["neutral", "joy", "surprise"]
            },
            "Samsung Galaxy & OnePlus Smartphones": {
                "category": "Smartphones",
                "brand": "Samsung / OnePlus",
                "semantic_profile": "Smartphone camera test, 5G mobile unboxing, hand holding phone, display review, mobile photography, fast charging battery gadget review.",
                "keywords": "flagship smartphone 5g mobile phone camera megapixel amoled screen unboxing mobile review battery backup processor snapdragon",
                "target_environments": ["tech_workspace", "outdoor_travel"],
                "target_visual_cues": ["phone", "cell phone", "holding", "camera", "hand"],
                "target_activities": ["unboxing", "taking photo", "reviewing", "scrolling"],
                "preferred_emotions": ["neutral", "joy", "surprise"]
            },

            # 🍕 Food, Cooking & Grocery
            "Zomato / Swiggy Food Delivery": {
                "category": "Food & Grocery",
                "brand": "Zomato / Swiggy",
                "semantic_profile": "Food on dining table, delicious hot meal, hungry people eating pizza burger biryani, kitchen recipe, ordering dinner snacks online at home.",
                "keywords": "food delivery eating pizza burger biryani hungry meal dinner lunch snack order food online swiggy zomato bhookh tasty dish",
                "target_environments": ["kitchen_dining"],
                "target_visual_cues": ["food", "plate", "table", "dish", "bowl", "pizza", "meal", "eating"],
                "target_activities": ["eating", "cooking", "ordering", "dining", "tasting"],
                "preferred_emotions": ["joy", "surprise", "neutral"]
            },
            "Blinkit / Zepto 10-Min Grocery": {
                "category": "Food & Grocery",
                "brand": "Blinkit / Zepto",
                "semantic_profile": "Kitchen grocery essentials, fresh vegetables milk bread, instant quick delivery in 10 minutes, emergency home kitchen supplies.",
                "keywords": "instant grocery delivery fresh fruits vegetables milk bread snacks 10 minutes quick delivery ghar ka samaan pantry",
                "target_environments": ["kitchen_dining"],
                "target_visual_cues": ["kitchen", "vegetables", "fruits", "table", "bottle", "refrigerator"],
                "target_activities": ["cooking", "preparing food", "shopping"],
                "preferred_emotions": ["neutral", "joy"]
            },

            # 🏏 Gaming, Esports & Fantasy
            "ASUS ROG & Razer Gaming Setup": {
                "category": "Gaming Gear",
                "brand": "ASUS ROG / Razer",
                "semantic_profile": "High performance PC battle station, mechanical RGB keyboard, pro esports gaming mouse, high fps graphics card streaming battle royale game.",
                "keywords": "pc gaming esports gaming mouse mechanical keyboard high fps streaming setup bgmi pubg free fire gta graphics card rtx",
                "target_environments": ["gaming_entertainment", "tech_workspace"],
                "target_visual_cues": ["screen", "monitor", "keyboard", "headphones", "gaming setup"],
                "target_activities": ["gaming", "streaming", "playing", "competing"],
                "preferred_emotions": ["surprise", "joy", "neutral"]
            },
            "Dream11 / My11Circle Fantasy Sports": {
                "category": "Fantasy Sports",
                "brand": "Dream11",
                "semantic_profile": "Live cricket sports stadium match, watching cricket score excitement, IPL tournament prediction, fantasy team selection winning cash prizes.",
                "keywords": "fantasy sports cricket match playing games predicting score ipl tournament match live commentary khelo jeeto cash team",
                "target_environments": ["gaming_entertainment", "outdoor_travel"],
                "target_visual_cues": ["stadium", "tv", "living room", "crowd", "field"],
                "target_activities": ["watching match", "cheering", "playing sports"],
                "preferred_emotions": ["joy", "surprise"]
            },

            # 🏃 Fitness, Gym & Wellness
            "Nike & Puma Sports Gear": {
                "category": "Fitness & Wellness",
                "brand": "Nike / Puma",
                "semantic_profile": "Athletic person working out, running outdoor marathon, gym sports shoes, active healthy training lifestyle, workout sneakers.",
                "keywords": "fitness running workout gym sports shoes active lifestyle athlete training sneakers exercise marathon sweat kasrat",
                "target_environments": ["gym_fitness", "outdoor_travel"],
                "target_visual_cues": ["shoes", "running", "gym", "workout", "person standing outside", "sports"],
                "target_activities": ["running", "jogging", "working out", "exercising", "training"],
                "preferred_emotions": ["joy", "neutral"]
            },
            "MuscleBlaze Whey Protein": {
                "category": "Fitness & Wellness",
                "brand": "MuscleBlaze",
                "semantic_profile": "Bodybuilding gym workout, lifting weights, muscle recovery, protein powder shake, fitness nutrition supplement.",
                "keywords": "whey protein body building gym workout muscle building bcaa creatine nutrition supplement protein powder shake",
                "target_environments": ["gym_fitness", "kitchen_dining"],
                "target_visual_cues": ["gym", "weights", "shaker", "bottle", "workout"],
                "target_activities": ["lifting", "working out", "drinking protein shake"],
                "preferred_emotions": ["neutral", "joy"]
            },

            # 📈 Finance & Fintech
            "Zerodha / Groww Stock Investing": {
                "category": "Finance",
                "brand": "Zerodha / Groww",
                "semantic_profile": "Stock market charts, financial growth, mutual funds SIP, trading portfolio demat account, investing money for future wealth.",
                "keywords": "stock market mutual funds investing money shares trading sip portfolio demat account stock analysis wealth share bazar",
                "target_environments": ["tech_workspace", "classroom_study"],
                "target_visual_cues": ["desk", "screen", "chart", "laptop", "office"],
                "target_activities": ["investing", "analyzing", "planning", "working"],
                "preferred_emotions": ["neutral", "joy"]
            },
            "CRED Credit Card Payments": {
                "category": "Finance",
                "brand": "CRED",
                "semantic_profile": "Premium lifestyle, credit card bill payments, cashback reward perks, financial credit score improvement, exclusive luxury offers.",
                "keywords": "credit card bills cashback reward payment app credit score money savings financial perks upi exclusive offers",
                "target_environments": ["tech_workspace", "outdoor_travel"],
                "target_visual_cues": ["phone", "luxury", "desk", "living room"],
                "target_activities": ["paying", "shopping", "managing finances"],
                "preferred_emotions": ["joy", "neutral"]
            },

            # 📚 Education & Upskilling
            "Scaler / Coursera Tech Upskilling": {
                "category": "Education & Coaching",
                "brand": "Scaler / Coursera",
                "semantic_profile": "Learning software engineering, web development coding bootcamp, python data science, computer career upskilling online courses.",
                "keywords": "software development data science python coding full stack web development tech career upskilling learn programming bootcamp ai engineer",
                "target_environments": ["classroom_study", "tech_workspace"],
                "target_visual_cues": ["laptop", "desk", "screen", "notes", "sitting at desk", "couch with laptop"],
                "target_activities": ["studying", "coding", "learning", "practicing"],
                "preferred_emotions": ["neutral", "joy"]
            },
            "Unacademy / Physics Wallah (PW)": {
                "category": "Education & Coaching",
                "brand": "Unacademy / PW",
                "semantic_profile": "Student exam preparation, JEE NEET coaching, classroom whiteboard teacher lecture, competitive exam syllabus mock test online classes.",
                "keywords": "exam preparation jee neet upsc online coaching student studies syllabus lectures mock test competitive exam padhai class",
                "target_environments": ["classroom_study"],
                "target_visual_cues": ["whiteboard", "classroom", "books", "notes", "teacher", "desk"],
                "target_activities": ["teaching", "studying", "taking notes", "learning"],
                "preferred_emotions": ["neutral", "joy"]
            },

            # ✈️ Travel & Tourism
            "MakeMyTrip / EaseMyTrip Holidays": {
                "category": "Travel & Holidays",
                "brand": "MakeMyTrip",
                "semantic_profile": "Scenic mountain vistas, airport flight booking, tropical beach holiday vacation tour package, explore travel destinations outdoors.",
                "keywords": "travel flight booking holiday vacation hotel stay tour package cheap flight tickets travelling abroad pahad trip tourism explore",
                "target_environments": ["outdoor_travel"],
                "target_visual_cues": ["mountain", "beach", "sky", "landscape", "hotel", "outdoor", "scenic"],
                "target_activities": ["traveling", "exploring", "walking outdoors", "vacationing"],
                "preferred_emotions": ["joy", "surprise", "neutral"]
            },
            "Airbnb Cozy Homestays": {
                "category": "Travel & Holidays",
                "brand": "Airbnb",
                "semantic_profile": "Cozy aesthetic villa homestay, mountain view cottage, weekend getaway vacation rental living room lounge with friends.",
                "keywords": "resort booking scenic villa stay cozy homestay mountain view beach vacation holiday rental weekend getaway luxury stay",
                "target_environments": ["outdoor_travel", "kitchen_dining"],
                "target_visual_cues": ["villa", "house", "mountain", "living room", "cozy", "scenic", "nature"],
                "target_activities": ["relaxing", "staying", "vacationing", "enjoying view"],
                "preferred_emotions": ["joy", "neutral"]
            },

            # 👗 Fashion & Beauty
            "Nykaa / Mamaearth Skincare": {
                "category": "Fashion & Beauty",
                "brand": "Nykaa / Mamaearth",
                "semantic_profile": "Skincare routine, glowing face makeup cosmetic beauty products, healthy skin hair care natural dermatology mirror tutorial.",
                "keywords": "skincare routine makeup glowing skin face wash sunscreen cosmetic beauty products hair care natural sunder twacha dermatology",
                "target_environments": ["tech_workspace", "kitchen_dining"],
                "target_visual_cues": ["face", "cosmetics", "bottle", "mirror", "person"],
                "target_activities": ["applying makeup", "skincare", "grooming", "smiling"],
                "preferred_emotions": ["joy", "neutral"]
            },
            "Myntra / Ajio Fashion Shopping": {
                "category": "Fashion & Beauty",
                "brand": "Myntra / Ajio",
                "semantic_profile": "Trendy stylish outfit, streetwear fashion clothing sale, festive dress shoes collection online shopping discount.",
                "keywords": "fashion outfit trendy clothes online shopping sale dresses streetwear shoes festive wedding wear kapde kharidna",
                "target_environments": ["outdoor_travel", "tech_workspace"],
                "target_visual_cues": ["clothes", "outfit", "dress", "fashion", "person standing", "shoes"],
                "target_activities": ["modeling", "shopping", "showing outfit", "walking"],
                "preferred_emotions": ["joy", "surprise"]
            },

            # 🚗 Automobile & Bikes
            "Tata Motors / Mahindra SUV Cars": {
                "category": "Automobile",
                "brand": "Tata / Mahindra",
                "semantic_profile": "Car test drive on highway road, SUV vehicle review, electric car mileage specs, road trip driving experience safety features.",
                "keywords": "car review electric vehicle mileage test drive safe suv automatic transmission top speed nayi gaadi road trip driving",
                "target_environments": ["automobile_transit", "outdoor_travel"],
                "target_visual_cues": ["car", "vehicle", "road", "steering wheel", "highway", "traffic", "driving"],
                "target_activities": ["driving", "riding", "test driving", "road tripping"],
                "preferred_emotions": ["neutral", "joy", "surprise"]
            },
            "Royal Enfield / Yamaha Cruiser Bikes": {
                "category": "Automobile",
                "brand": "Royal Enfield / Yamaha",
                "semantic_profile": "Motorcycle road trip, cruiser bike exhaust sound, biker helmet riding gear on scenic highway mountain pass.",
                "keywords": "motorcycle ride biking road trip exhaust sound cruise bike rider helmet superbike bike review 350cc riding gear",
                "target_environments": ["automobile_transit", "outdoor_travel"],
                "target_visual_cues": ["motorcycle", "bike", "helmet", "road", "highway", "rider"],
                "target_activities": ["biking", "riding", "cruising"],
                "preferred_emotions": ["joy", "surprise", "neutral"]
            }
        }
        
        # Pre-compute Dense Vectors for all Ad Campaigns
        self.ad_vectors = {}
        for ad_name, data in self.ad_campaigns.items():
            enriched_desc = f"{data['semantic_profile']} Keywords: {data['keywords']}"
            vec = self.ad_model.encode(enriched_desc, normalize_embeddings=True)
            self.ad_vectors[ad_name] = torch.tensor(vec, device=self.device)
            
        print(f"✅ Loaded {len(self.ad_campaigns)} Scene-Aware Ad Campaign Profiles!")
        print("=" * 60)

    def recommend_for_video(self, video_id, threshold=0.38, top_k_per_campaign=1):
        """
        Executes Hybrid Search (Dense Semantic + Sparse BM25 + Scene Context Affinity)
        and recommends contextually fitting ads based on the holistic scene environment.
        """
        video_id = clean_video_id(video_id)
        json_path = os.path.join(self.scenes_dir, f"{video_id}_scenes.json")
        npy_path = os.path.join(self.scenes_dir, f"{video_id}_embeddings.npy")
        
        if not os.path.exists(json_path) or not os.path.exists(npy_path):
            print(f"❌ Scene data for {video_id} not found. Please run scene_analyzer.py first.")
            print(f"   Missing: {json_path} or {npy_path}")
            return []

        # 1. Load Video & Scene Data
        with open(json_path, 'r', encoding='utf-8') as f:
            video_data = json.load(f)
        
        scenes = video_data.get('scenes', [])
        embeddings_matrix = np.load(npy_path)
        
        if len(embeddings_matrix) == 0 or len(scenes) == 0:
            print(f"⚠️ Video {video_id} has no scene records or embeddings.")
            return []

        video_tensor = torch.tensor(embeddings_matrix, device=self.device)
        total_scenes = len(scenes)

        print(f"\n🎬 Multi-Modal Ad Matching for Video ID: {video_id}")
        video_title = video_data.get('metadata', {}).get('title', 'Unknown Title')
        print(f"📹 Video Title: \"{video_title}\"")
        print(f"🎞️ Total Analyzed Scenes: {total_scenes}")
        print("=" * 60)

        # 2. Build Sparse BM25 Corpus from Scene Profiles
        scene_corpus = []
        for s in scenes:
            caption = s.get('visual_caption', '')
            dialogue = s.get('transcript_dialogue', '')
            emotion = s.get('emotion_tag', '')
            # Weight visual caption and dialogue heavily in BM25 document representation
            doc_text = f"Visuals: {caption} {caption} Spoken: {dialogue} Emotion: {emotion}"
            scene_corpus.append(doc_text)

        bm25_retriever = BM25Retriever(scene_corpus)

        # 3. Multi-Modal Hybrid Search per Campaign
        recommendations = []

        for ad_name, campaign in self.ad_campaigns.items():
            ad_vector = self.ad_vectors[ad_name]
            
            # --- A. Dense Semantic Vector Retrieval ---
            dense_similarities = F.cosine_similarity(ad_vector.unsqueeze(0), video_tensor).cpu().numpy()
            
            # --- B. Sparse BM25 Lexical Retrieval ---
            query_bm25 = f"{campaign['keywords']} {campaign['semantic_profile']}"
            sparse_bm25_scores = bm25_retriever.get_scores(query_bm25)
            
            # --- C. Scene Context & Setting Affinity ---
            scene_affinity_scores = np.zeros(total_scenes, dtype=np.float32)
            for idx, scene in enumerate(scenes):
                vis = scene.get('visual_caption', '')
                dial = scene.get('transcript_dialogue', '')
                emo = scene.get('emotion_tag', 'neutral')
                scene_affinity_scores[idx] = SceneContextClassifier.compute_scene_affinity(vis, dial, emo, campaign)
            
            # --- D. Calibrated Hybrid Fusion ---
            # Hybrid Score = 45% Dense + 35% Sparse BM25 + 20% Scene Context Affinity
            raw_hybrid_scores = (
                0.45 * dense_similarities + 
                0.35 * sparse_bm25_scores + 
                0.20 * scene_affinity_scores
            )

            # Apply Context Consistency Gate (prevents single polysemous word hits like 'apple fruit' triggering tech ads)
            hybrid_scores = np.copy(raw_hybrid_scores)
            for idx in range(total_scenes):
                d_val = dense_similarities[idx]
                aff_val = scene_affinity_scores[idx]
                if d_val < 0.28 and aff_val < 0.10:
                    # Penalize lexical-only outliers when neither semantic nor scene setting agrees
                    hybrid_scores[idx] *= 0.60

            # Find best scene candidate(s)
            best_scene_indices = np.argsort(-hybrid_scores)[:top_k_per_campaign]
            
            for best_idx in best_scene_indices:
                final_score = float(hybrid_scores[best_idx])
                dense_score = float(dense_similarities[best_idx])
                sparse_score = float(sparse_bm25_scores[best_idx])
                affinity_score = float(scene_affinity_scores[best_idx])
                
                # Check confidence threshold
                if final_score >= threshold:
                    scene_info = scenes[best_idx]
                    timestamp_sec = scene_info.get('start_sec', 0.0)
                    mins, secs = divmod(int(timestamp_sec), 60)
                    time_fmt = f"{mins:02d}:{secs:02d}"
                    
                    dialogue = scene_info.get('transcript_dialogue', 'None')
                    caption = scene_info.get('visual_caption', 'N/A')
                    mood = scene_info.get('emotion_tag', 'neutral')
                    
                    # Generate natural language rationale
                    reasons = []
                    if caption and caption != "N/A":
                        reasons.append(f"Scene shows: '{caption}'")
                    if affinity_score > 0.2:
                        reasons.append("High environment & activity alignment")
                    if dense_score > 0.4:
                        reasons.append("Strong semantic context match")
                    if sparse_score > 0.3:
                        reasons.append("Direct keyword relevance")
                    
                    rationale = " | ".join(reasons) if reasons else "Multi-modal contextual match"

                    rec = {
                        "ad_name": ad_name,
                        "brand": campaign.get("brand", ""),
                        "category": campaign.get("category", ""),
                        "scene_id": scene_info.get("scene_id", best_idx + 1),
                        "timestamp_sec": timestamp_sec,
                        "timestamp_formatted": time_fmt,
                        "hybrid_score": round(final_score, 4),
                        "relevance_percentage": round(final_score * 100, 1),
                        "dense_score": round(dense_score, 3),
                        "sparse_score": round(sparse_score, 3),
                        "scene_affinity": round(affinity_score, 3),
                        "visual_caption": caption,
                        "scene_mood": mood,
                        "context_dialogue": dialogue,
                        "rationale": rationale
                    }
                    recommendations.append(rec)

        # Sort recommendations by highest hybrid relevance score
        recommendations.sort(key=lambda x: x['hybrid_score'], reverse=True)

        # Display formatted results
        if not recommendations:
            print(f"ℹ️ No ad placements matched above the hybrid relevance threshold ({threshold*100:.0f}%).")
        else:
            print(f"✨ Found {len(recommendations)} Context-Aware Ad Placements (Hybrid Relevance >= {threshold*100:.0f}%):\n")
            for idx, rec in enumerate(recommendations, 1):
                preview_dial = rec['context_dialogue'][:75] + ("..." if len(rec['context_dialogue']) > 75 else "")
                print(f"[{idx}] 🎯 Ad: [{rec['ad_name']}] ({rec['brand']})")
                print(f"    ⏱️  Timestamp: {rec['timestamp_formatted']} (Scene #{rec['scene_id']} @ {rec['timestamp_sec']}s)")
                print(f"    📊 Hybrid Relevance: {rec['relevance_percentage']}% (Dense: {rec['dense_score']*100:.1f}%, BM25: {rec['sparse_score']*100:.1f}%, Scene Setting: {rec['scene_affinity']*100:.1f}%)")
                print(f"    🖼️  Visual Scene: \"{rec['visual_caption']}\"")
                print(f"    🎭 Scene Mood: {rec['scene_mood']}")
                print(f"    💬 Spoken Dialogue: \"{preview_dial}\"")
                print(f"    💡 Placement Rationale: {rec['rationale']}")
                print("-" * 60)

        return recommendations


# =====================================================================
# 4. CLI ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        vid_input = sys.argv[1]
    else:
        vid_input = input("Enter Video ID or YouTube URL to get hybrid scene recommendations (e.g., AiDyP-Q4Kgk): ").strip()
        
    recommender = AdvancedAdRecommender()
    recommender.recommend_for_video(vid_input)