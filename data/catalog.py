"""Benchmark music catalog populated with psycho-acoustic descriptors for AffectiveEcho."""
from typing import List
from src.audio.mood_acoustic_mapping import AcousticAffectMapper, TrackAffectProfile


def get_benchmark_catalog() -> List[TrackAffectProfile]:
    mapper = AcousticAffectMapper()

    raw_tracks = [
        # --- Quadrant 1: High Valence, High Arousal (Joy / Euphoria / Celebration) ---
        ("TRK_001", "Badtameez Dil", "Benny Dayal", "Bollywood Dance", {"tempo": 150.0, "rms": 0.28, "spectral_centroid": 3600.0, "chroma_mean": 0.52, "zero_crossing_rate": 0.09}),
        ("TRK_002", "Matargashti", "Mohit Chauhan", "Bollywood Pop", {"tempo": 138.0, "rms": 0.22, "spectral_centroid": 3100.0, "chroma_mean": 0.48, "zero_crossing_rate": 0.07}),
        ("TRK_003", "Happy Hours", "Mika Singh", "Bollywood Upbeat", {"tempo": 142.0, "rms": 0.25, "spectral_centroid": 3300.0, "chroma_mean": 0.50, "zero_crossing_rate": 0.08}),
        ("TRK_004", "Gallan Goodiyaan", "Yashita Sharma", "Bhangra Pop", {"tempo": 160.0, "rms": 0.30, "spectral_centroid": 3800.0, "chroma_mean": 0.54, "zero_crossing_rate": 0.10}),
        ("TRK_005", "Uptown Funk", "Mark Ronson ft. Bruno Mars", "Funk Pop", {"tempo": 115.0, "rms": 0.26, "spectral_centroid": 3400.0, "chroma_mean": 0.51, "zero_crossing_rate": 0.08}),
        ("TRK_006", "Kar Gayi Chull", "Badshah, Neha Kakkar", "Club Dance", {"tempo": 128.0, "rms": 0.29, "spectral_centroid": 3500.0, "chroma_mean": 0.49, "zero_crossing_rate": 0.09}),
        ("TRK_007", "Can't Stop the Feeling!", "Justin Timberlake", "Disco Pop", {"tempo": 113.0, "rms": 0.24, "spectral_centroid": 3200.0, "chroma_mean": 0.53, "zero_crossing_rate": 0.07}),
        ("TRK_008", "Aankh Marey", "Neha Kakkar, Mika", "Remix Dance", {"tempo": 140.0, "rms": 0.28, "spectral_centroid": 3700.0, "chroma_mean": 0.52, "zero_crossing_rate": 0.09}),

        # --- Quadrant 2: Low Valence, High Arousal (Anger / Agitation / Frustration) ---
        ("TRK_009", "Junoon", "Ali Azmat", "Sufi Rock", {"tempo": 148.0, "rms": 0.27, "spectral_centroid": 2900.0, "chroma_mean": 0.28, "zero_crossing_rate": 0.11}),
        ("TRK_010", "Dushman Na Kare", "Amit Kumar", "Dramatic Rock", {"tempo": 132.0, "rms": 0.24, "spectral_centroid": 2700.0, "chroma_mean": 0.25, "zero_crossing_rate": 0.09}),
        ("TRK_011", "Thukra Ke", "Yasser Desai", "Aggressive Electro", {"tempo": 136.0, "rms": 0.26, "spectral_centroid": 2850.0, "chroma_mean": 0.26, "zero_crossing_rate": 0.10}),
        ("TRK_012", "Apna Time Aayega", "Ranveer Singh", "Hard Hip-Hop", {"tempo": 144.0, "rms": 0.29, "spectral_centroid": 3100.0, "chroma_mean": 0.24, "zero_crossing_rate": 0.12}),
        ("TRK_013", "Sadda Haq", "Mohit Chauhan", "Alternative Rock", {"tempo": 140.0, "rms": 0.31, "spectral_centroid": 3200.0, "chroma_mean": 0.27, "zero_crossing_rate": 0.11}),
        ("TRK_014", "In The End", "Linkin Park", "Nu Metal", {"tempo": 105.0, "rms": 0.25, "spectral_centroid": 2600.0, "chroma_mean": 0.26, "zero_crossing_rate": 0.08}),

        # --- Quadrant 3: Low Valence, Low Arousal (Sadness / Melancholy / Solitude) ---
        ("TRK_015", "Tujhe Bhula Diya", "Mohit Chauhan, Shekhar", "Sad Ballad", {"tempo": 84.0, "rms": 0.09, "spectral_centroid": 1450.0, "chroma_mean": 0.23, "zero_crossing_rate": 0.03}),
        ("TRK_016", "Do Pal", "Lata Mangeshkar, Sonu Nigam", "Melancholy Classic", {"tempo": 78.0, "rms": 0.07, "spectral_centroid": 1300.0, "chroma_mean": 0.25, "zero_crossing_rate": 0.025}),
        ("TRK_017", "Main Jahan Rahoon", "Rahat Fateh Ali Khan", "Sufi Ballad", {"tempo": 82.0, "rms": 0.10, "spectral_centroid": 1500.0, "chroma_mean": 0.26, "zero_crossing_rate": 0.03}),
        ("TRK_018", "Channa Mereya", "Arijit Singh", "Acoustic Lament", {"tempo": 80.0, "rms": 0.08, "spectral_centroid": 1400.0, "chroma_mean": 0.24, "zero_crossing_rate": 0.032}),
        ("TRK_019", "Agar Tum Saath Ho", "Alka Yagnik, Arijit Singh", "Emotional Ballad", {"tempo": 86.0, "rms": 0.11, "spectral_centroid": 1600.0, "chroma_mean": 0.25, "zero_crossing_rate": 0.035}),
        ("TRK_020", "Someone Like You", "Adele", "Piano Ballad", {"tempo": 68.0, "rms": 0.06, "spectral_centroid": 1250.0, "chroma_mean": 0.27, "zero_crossing_rate": 0.022}),
        ("TRK_021", "Fix You", "Coldplay", "Atmospheric Sad", {"tempo": 70.0, "rms": 0.08, "spectral_centroid": 1380.0, "chroma_mean": 0.28, "zero_crossing_rate": 0.028}),

        # --- Quadrant 4: High Valence, Low Arousal (Peace / Calm / Serenity / Relaxation) ---
        ("TRK_022", "Kun Faya Kun", "A.R. Rahman, Javed Ali", "Spiritual Ambient", {"tempo": 90.0, "rms": 0.11, "spectral_centroid": 2100.0, "chroma_mean": 0.44, "zero_crossing_rate": 0.04}),
        ("TRK_023", "Theher Ja", "Armaan Malik", "Lo-Fi Acoustic", {"tempo": 94.0, "rms": 0.10, "spectral_centroid": 2000.0, "chroma_mean": 0.42, "zero_crossing_rate": 0.038}),
        ("TRK_024", "Mitwa", "Shafqat Amanat Ali", "Smooth Melodic", {"tempo": 98.0, "rms": 0.13, "spectral_centroid": 2300.0, "chroma_mean": 0.46, "zero_crossing_rate": 0.042}),
        ("TRK_025", "Iktara", "Kavita Seth, Tochi Raina", "Folk Ambient", {"tempo": 92.0, "rms": 0.10, "spectral_centroid": 1950.0, "chroma_mean": 0.43, "zero_crossing_rate": 0.035}),
        ("TRK_026", "Weightless", "Marconi Union", "Ambient Therapy", {"tempo": 60.0, "rms": 0.05, "spectral_centroid": 1600.0, "chroma_mean": 0.40, "zero_crossing_rate": 0.02}),
        ("TRK_027", "Clair de Lune", "Claude Debussy", "Classical Serene", {"tempo": 65.0, "rms": 0.04, "spectral_centroid": 1400.0, "chroma_mean": 0.45, "zero_crossing_rate": 0.018}),

        # --- Neutral / Focus (Balanced Center) ---
        ("TRK_028", "Jee Le Zara", "Vishal Dadlani", "Contemporary Pop", {"tempo": 110.0, "rms": 0.15, "spectral_centroid": 2400.0, "chroma_mean": 0.36, "zero_crossing_rate": 0.05}),
        ("TRK_029", "Hello", "Lionel Richie", "Soft Rock", {"tempo": 102.0, "rms": 0.14, "spectral_centroid": 2200.0, "chroma_mean": 0.35, "zero_crossing_rate": 0.045}),
        ("TRK_030", "Paisa Paisa", "RDB", "Mid-tempo Dance", {"tempo": 118.0, "rms": 0.18, "spectral_centroid": 2600.0, "chroma_mean": 0.37, "zero_crossing_rate": 0.06})
    ]

    profiles = []
    for tid, title, artist, genre, feats in raw_tracks:
        prof = mapper.create_track_profile(tid, title, artist, genre, feats)
        profiles.append(prof)

    return profiles
