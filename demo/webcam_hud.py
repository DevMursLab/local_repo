"""Interactive Real-time Affective Music Recommender Demonstration."""
import cv2
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import AffectiveEchoPipeline
from data.catalog import get_benchmark_catalog


def run_hud_demo():
    print("=" * 65)
    print("   AffectiveEcho: Real-Time Affective Music Recommender HUD")
    print("=" * 65)
    print("[*] Initializing Pipeline...")
    catalog = get_benchmark_catalog()
    pipeline = AffectiveEchoPipeline(music_catalog=catalog)

    print("[*] Opening video capture (Webcam 0)...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[!] No webcam found. Running synthetic interactive video stream mode.")
        synthetic_mode = True
    else:
        synthetic_mode = False
        print("[+] Webcam active! Press 'q' in video window to exit.")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            if synthetic_mode:
                # Generate synthetic test frame with simulated smiling face
                frame = np.full((480, 640, 3), 40, dtype=np.uint8)
                cv2.circle(frame, (320, 240), 120, (180, 200, 220), -1)
                cv2.circle(frame, (280, 210), 15, (50, 50, 50), -1)
                cv2.circle(frame, (360, 210), 15, (50, 50, 50), -1)
                # smiling arc
                cv2.ellipse(frame, (320, 260), (45, 30), 0, 0, 180, (40, 40, 180), 5)
            else:
                ret, frame = cap.read()
                if not ret:
                    break

            # Process frame through AffectiveEcho Pipeline
            result = pipeline.process_frame(frame, top_k=3)

            # Draw HUD Overlays
            hud_frame = frame.copy()
            if result.get("face_detected"):
                x1, y1, x2, y2 = result["bbox"]
                cv2.rectangle(hud_frame, (x1, y1), (x2, y2), (0, 255, 120), 2)

                # Display Emotion and Valence-Arousal radar
                lbl = result["discrete_emotion"]
                conf = result["confidence"]
                val = result["valence"]
                aro = result["arousal"]
                fps = result["fps"]

                cv2.putText(hud_frame, f"Emotion: {lbl} ({conf*100:.1f}%)", (x1, max(30, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2)
                cv2.putText(hud_frame, f"Valence: {val:+.2f} | Arousal: {aro:+.2f}", (x1, y2 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 50), 2)
                cv2.putText(hud_frame, f"FPS: {fps:.1f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

                # Overlay recommended playlist box
                y_offset = 80
                cv2.putText(hud_frame, "=== AffectiveEcho Recommended Tracks ===", (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                for rec in result["recommendations"]:
                    y_offset += 26
                    rec_txt = f"{rec.rank}. {rec.title} - {rec.artist} (Target Val: {rec.target_step_valence:+.2f})"
                    cv2.putText(hud_frame, rec_txt, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 255, 255), 1)
            else:
                cv2.putText(hud_frame, "Scanning for face...", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            frame_count += 1
            if synthetic_mode:
                print(f"[Iter {frame_count}] Detected: {result.get('discrete_emotion')} | Val: {result.get('valence'):+.2f} | Aro: {result.get('arousal'):+.2f} | Latency: {result.get('latency_ms')} ms")
                time.sleep(0.5)
                if frame_count >= 5:
                    print("[+] Synthetic interactive run completed successfully!")
                    break
            else:
                cv2.imshow("AffectiveEcho Real-Time HUD", hud_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        if not synthetic_mode:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    run_hud_demo()
