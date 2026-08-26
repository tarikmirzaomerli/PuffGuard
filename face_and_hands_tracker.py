import cv2
import mediapipe as mp
import math
import time
import os
import threading
import numpy as np
from datetime import datetime
from collections import deque
from ultralytics import YOLO

# Modern Windows Toast Bildirimi
try:
    from win11toast import toast
    USE_WIN11TOAST = True
except Exception:
    USE_WIN11TOAST = False
    from plyer import notification

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Proje Kok Dizini
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

def send_desktop_notification(title, message):
    """Masaustu bildirimini arka planda asenkron gonderir."""
    def _notify():
        try:
            if USE_WIN11TOAST:
                toast(title, message, duration="short")
            else:
                notification.notify(title=title, message=message, timeout=4)
        except Exception as e:
            print(f"[!] Bildirim hatasi: {e}")

    threading.Thread(target=_notify, daemon=True).start()

def save_video_async(frames_list, video_path, width, height, fps=30.0):
    """300 karelik (5sn oncesi + 5sn sonrasi) tamponu arka planda MP4 olarak kaydeder."""
    def _save():
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            for f in frames_list:
                out.write(f)
            out.release()
            print(f"\n[+] 10 SANIYELIK VIDEO (5sn Once + 5sn Sonra) KAYDEDILDI -> {video_path}")
        except Exception as e:
            print(f"[!] Video kaydetme hatasi: {e}")

    threading.Thread(target=_save, daemon=True).start()

def load_roboflow_yolo():
    """Model agirligini proje dizininden yukler."""
    print("==================================================")
    print("[+] Model yukleniyor...")
    model_path = os.path.join(PROJECT_DIR, "best.pt")

    if not os.path.exists(model_path):
        model_path = "best.pt"

    if os.path.exists(model_path):
        print(f"[+] YOLO modeli yukleniyor: '{model_path}'")
        model = YOLO(model_path)
        print(f"[+] Model basariyla yuklendi! Siniflar: {model.names}")
        print("==================================================")
        return model
    else:
        print("[!] 'best.pt' modeli bulunamadi!")
        return None

def main():
    # 1. Proje Icindeki 'cigarettes-foto' Klasorunu Olustur / Kontrol Et
    photo_dir = os.path.join(PROJECT_DIR, 'cigarettes-foto')
    if not os.path.exists(photo_dir):
        os.makedirs(photo_dir, exist_ok=True)
        print(f"[+] Kayit klasoru olusturuldu: {photo_dir}")
    else:
        print(f"[+] Kayit klasoru hazir: {photo_dir}")

    # 2. YOLO Modelini Yukleme
    yolo_model = load_roboflow_yolo()

    # 3. MediaPipe Modulleri
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Hata: Kamera acilamadi!")
        send_desktop_notification(
            "UYARI: Kamera Açılamadı!",
            "Kamera cihazına erişilemiyor veya başka bir uygulama kullanıyor."
        )
        return

    # Kamera cozunurlugu 640x480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # --- PARAMETRELER ---
    MOUTH_CROP_SIZE = 150                      # Agiz kesme boyutu (150x150 px)
    CONFIDENCE_THRESHOLD = 0.65                # Guven Esigi (%65)
    REQUIRED_CONSECUTIVE_FRAMES = 12           # 12 Kesintisiz Kare Dogrulamasi
    ALLOWED_CLASSES = {"cigarette"}            # Hedef sinif
    
    # --- COOLDOWN SURELERI ---
    CIGARETTE_NOTIFICATION_COOLDOWN = 900.0    # 15 DAKIKA (900 saniye) Cooldown
    SECURITY_NOTIFICATION_COOLDOWN = 10.0      # Kamera engel/kayip icin 10 saniye

    # --- KAGIT / KARARTMA / ENGEL TESPIT PARAMETRELERI ---
    BLOCK_DURATION_REQ = 1.5         # 1.5 saniye kesintisiz engelleme
    LAPLACIAN_VAR_THRESHOLD = 40.0   # Doku/Kenar keskinligi
    STD_DEV_THRESHOLD = 18.0         # Renk homojenligi
    DARK_THRESHOLD = 40.0            # Karanlik/Siyah esigi

    # 10 SANIYELIK (300 KARE) SUREKLI DONEN TAMPON
    frame_buffer = deque(maxlen=300)

    # 5 SANIYE ONCESI + 5 SANIYE SONRASI KAYIT DEGISKENLERI
    POST_EVENT_FRAMES_REQUIRED = 150  # Tespitten sonra 150 kare (5 saniye) daha topla
    recording_post_event = False
    post_event_counter = 0
    event_timestamp_str = ""

    # Durum ve sayac degiskenleri
    consecutive_detection_count = 0
    last_cigarette_notification_time = 0.0     # 15 dakikalik zaman damgasi
    last_camera_loss_time = 0.0
    last_blocked_notification_time = 0.0
    block_start_time = None
    last_valid_frame = None
    last_saved_video_name = ""

    frame_count = 0
    fps_start_time = time.time()
    fps = 0

    # Thread / Asenkron degiskenler
    is_yolo_busy = False
    detected_objects = []
    last_seen_time = 0.0

    lip_indices = set()
    for conn in mp_face_mesh.FACEMESH_LIPS:
        lip_indices.add(conn[0])
        lip_indices.add(conn[1])

    def run_yolo_on_mouth_roi(roi_img, offset_x, offset_y):
        nonlocal is_yolo_busy, detected_objects, last_seen_time
        try:
            results = yolo_model(roi_img, conf=CONFIDENCE_THRESHOLD, verbose=False, imgsz=160)
            current_boxes = []

            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = r.names[cls_id].strip().lower() if (hasattr(r, 'names') and cls_id in r.names) else "cigarette"

                    if (cls_name in ALLOWED_CLASSES or cls_id == 0) and conf >= CONFIDENCE_THRESHOLD:
                        bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                        fx1 = offset_x + bx1
                        fy1 = offset_y + by1
                        fx2 = offset_x + bx2
                        fy2 = offset_y + by2
                        current_boxes.append((fx1, fy1, fx2, fy2, cls_name, conf))
                        last_seen_time = time.time()
                        print(f"[*] SIGARA YAKALANDI: %{int(conf*100)}")

            detected_objects = current_boxes
        except Exception as e:
            pass
        finally:
            is_yolo_busy = False

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh, mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        print("==================================================")
        print("Sigara Takip Sistemi Aktif.")
        print(f"- Proje Klasoru: {PROJECT_DIR}")
        print(f"- Kayit Klasoru: {photo_dir}")
        print(f"- Video Yapisi: 5sn Once + 5sn Sonra (Toplam 10sn / 300 Kare)")
        print(f"- Sigara Cooldown: 15 Dakika ({int(CIGARETTE_NOTIFICATION_COOLDOWN)}s)")
        print("- Cikis: 'q' tusu")
        print("==================================================")

        while True:
            current_time = time.time()
            success, frame = cap.read()

            # --- 1. GORUNTU KAYBI (FRAME LOSS) ---
            if not success or frame is None:
                if (current_time - last_camera_loss_time) >= SECURITY_NOTIFICATION_COOLDOWN:
                    print("\n[!] UYARI: Kamera Bağlantısı Kesildi!")
                    if last_valid_frame is not None:
                        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        closed_photo_name = f"camera_closed_{now_str}.jpg"
                        closed_photo_path = os.path.join(photo_dir, closed_photo_name)
                        cv2.imwrite(closed_photo_path, last_valid_frame)
                        print(f"[+] Kapanmadan onceki kare kaydedildi: {closed_photo_path}")

                    send_desktop_notification(
                        "UYARI: Kamera Bağlantısı Kesildi!",
                        "Kamera görüntüsü alınamıyor. Son kare kaydedildi."
                    )
                    last_camera_loss_time = current_time

                time.sleep(0.1)
                continue

            last_valid_frame = frame.copy()
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # 300 KARELIK TAMPONA SUREKLI EKLE
            frame_buffer.append(frame.copy())

            # --- 5 SANIYE SONRASI KARE TOPLAMA VE KAYDI TAMAMLAMA ---
            if recording_post_event:
                post_event_counter += 1
                if post_event_counter >= POST_EVENT_FRAMES_REQUIRED:
                    video_name = f"cigarette_video_10s_{event_timestamp_str}.mp4"
                    video_path = os.path.join(photo_dir, video_name)

                    # Asenkron MP4 olarak kaydet
                    save_video_async(list(frame_buffer), video_path, w, h, fps=30.0)
                    last_saved_video_name = video_name

                    # Bildirimi gonder
                    send_desktop_notification(
                        "Uyarı: Sigara Doğrulandı!",
                        f"5 sn öncesi ve 5 sn sonrasını içeren 10 saniyelik video '{video_name}' kaydedildi."
                    )

                    # 15 Dakikalik cooldown baslat ve bayragi sifirla
                    last_cigarette_notification_time = time.time()
                    recording_post_event = False
                    post_event_counter = 0
                    print(f"[i] 5sn oncesi + 5sn sonrasi video kaydi tamamlandi. 15 dakikalik bekleme basladi.")

            # FPS Sayaci
            frame_count += 1
            if time.time() - fps_start_time >= 1.0:
                fps = frame_count
                frame_count = 0
                fps_start_time = time.time()

            # RGB Formati
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False

            face_results = face_mesh.process(rgb_frame)
            hand_results = hands.process(rgb_frame)

            rgb_frame.flags.writeable = True

            # --- 2. AKILLI ENGEL / KAGIT / KARARTMA TESPITI ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avg_brightness = float(np.mean(gray))
            std_brightness = float(np.std(gray))
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            has_face = bool(face_results.multi_face_landmarks)

            is_camera_obstructed = False
            block_reason = ""

            if avg_brightness < DARK_THRESHOLD:
                is_camera_obstructed = True
                block_reason = "Karanlik / Karartma"
            elif (laplacian_var < LAPLACIAN_VAR_THRESHOLD or std_brightness < STD_DEV_THRESHOLD) and not has_face:
                is_camera_obstructed = True
                block_reason = "Kagit / Fiziksel Engel"

            if is_camera_obstructed:
                if block_start_time is None:
                    block_start_time = current_time

                block_duration = current_time - block_start_time

                if block_duration >= BLOCK_DURATION_REQ:
                    if (current_time - last_blocked_notification_time) >= SECURITY_NOTIFICATION_COOLDOWN:
                        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        blocked_photo_name = f"camera_blocked_{now_str}.jpg"
                        blocked_photo_path = os.path.join(photo_dir, blocked_photo_name)
                        cv2.imwrite(blocked_photo_path, frame)
                        print(f"\n[!] UYARI: Kamera Görüşü Engellendi ({block_reason})! Fotoğraf: {blocked_photo_name}")

                        send_desktop_notification(
                            "UYARI: Kamera Görüşü Engellendi!",
                            f"Kamera görüşü {block_duration:.1f} saniyedir engellendi ({block_reason}). Kanıt kaydedildi."
                        )
                        last_blocked_notification_time = current_time

                cv2.rectangle(frame, (20, h // 2 - 40), (w - 20, h // 2 + 40), (0, 0, 180), -1)
                cv2.putText(frame, f"UYARI: KAMERA GORUSU ENGELLENDI ({block_reason}) {block_duration:.1f}s", (35, h // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            else:
                block_start_time = None

            lip_center = None
            detected_hands = []

            # 3. Dudak Landmarklari & Dudak Merkezi
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_LIPS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                    )

                    lip_x = []
                    lip_y = []
                    for idx in lip_indices:
                        lm = face_landmarks.landmark[idx]
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lip_x.append(cx)
                        lip_y.append(cy)

                    if lip_x and lip_y:
                        lip_center = (int(sum(lip_x) / len(lip_x)), int(sum(lip_y) / len(lip_y)))
                        cv2.circle(frame, lip_center, 4, (255, 0, 0), cv2.FILLED)

            # 4. El Landmarklari
            if hand_results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=1, circle_radius=1),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(180, 180, 180), thickness=1)
                    )

                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                    hand_label = handedness.classification[0].label
                    detected_hands.append(((ix, iy), hand_label))

                    cv2.circle(frame, (ix, iy), 7, (0, 255, 0), cv2.FILLED)

            # 5. Dudak Merkezli 150x150 ROI Kirpma & YOLO Inference
            if lip_center is not None and not is_camera_obstructed:
                half_sz = MOUTH_CROP_SIZE // 2
                rx1 = max(0, lip_center[0] - half_sz)
                ry1 = max(0, lip_center[1] - half_sz)
                rx2 = min(w, rx1 + MOUTH_CROP_SIZE)
                ry2 = min(h, ry1 + MOUTH_CROP_SIZE)

                if rx2 - rx1 < MOUTH_CROP_SIZE and rx1 > 0:
                    rx1 = max(0, rx2 - MOUTH_CROP_SIZE)
                if ry2 - ry1 < MOUTH_CROP_SIZE and ry1 > 0:
                    ry1 = max(0, ry2 - MOUTH_CROP_SIZE)

                mouth_crop = frame[ry1:ry2, rx1:rx2]

                if yolo_model is not None and mouth_crop.size > 0 and not is_yolo_busy:
                    is_yolo_busy = True
                    threading.Thread(target=run_yolo_on_mouth_roi, args=(mouth_crop.copy(), rx1, ry1), daemon=True).start()

                # Agiz ROI Cercevesi
                has_cig = len(detected_objects) > 0
                roi_box_color = (0, 0, 255) if has_cig else (255, 255, 0)
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), roi_box_color, 2)
                cv2.putText(frame, "Agiz ROI 150x150", (rx1, max(15, ry1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, roi_box_color, 1)

            # 6. Dogrulama Sayaci (12 Kare)
            is_currently_detected = len(detected_objects) > 0 or (current_time - last_seen_time < 0.35)
            best_score = 0.0

            if is_currently_detected and not is_camera_obstructed:
                consecutive_detection_count += 1
                for fx1, fy1, fx2, fy2, label, conf in detected_objects:
                    if conf > best_score:
                        best_score = conf

                    display_text = f"SIGARA %{int(conf*100)}"
                    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 0, 255), 3)
                    (tw, th), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(frame, (fx1, max(0, fy1 - 22)), (fx1 + tw + 8, max(22, fy1)), (0, 0, 255), -1)
                    cv2.putText(frame, display_text, (fx1 + 4, max(16, fy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            else:
                consecutive_detection_count = 0

            # 7. 15 Dakikalik Cooldown ve 5sn Once + 5sn Sonra Kayit Mantigi
            cooldown_remaining = max(0.0, CIGARETTE_NOTIFICATION_COOLDOWN - (current_time - last_cigarette_notification_time))
            in_cooldown = (current_time - last_cigarette_notification_time) < CIGARETTE_NOTIFICATION_COOLDOWN
            is_confirmed = consecutive_detection_count >= REQUIRED_CONSECUTIVE_FRAMES

            # Sag ust sayac paneli
            panel_w, panel_h = 300, 85
            panel_x = w - panel_w - 10
            panel_y = 10

            overlay = frame.copy()
            bg_col = (0, 0, 200) if is_confirmed else ((0, 100, 200) if consecutive_detection_count > 0 else (40, 40, 40))
            cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), bg_col, -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
            cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (255, 255, 255), 1)

            cv2.putText(frame, "SIGARA DOGRULAMA SAYACI", (panel_x + 10, panel_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
            
            counter_str = f"Kare: {consecutive_detection_count} / {REQUIRED_CONSECUTIVE_FRAMES} (conf >= %{int(CONFIDENCE_THRESHOLD*100)})"
            cv2.putText(frame, counter_str, (panel_x + 10, panel_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)

            # Ilerleme cubugu
            prog_ratio = min(1.0, consecutive_detection_count / REQUIRED_CONSECUTIVE_FRAMES)
            bar_w = int((panel_w - 20) * prog_ratio)
            bar_color = (0, 255, 0) if is_confirmed else (0, 255, 255)
            cv2.rectangle(frame, (panel_x + 10, panel_y + 55), (panel_x + panel_w - 10, panel_y + 68), (60, 60, 60), -1)
            cv2.rectangle(frame, (panel_x + 10, panel_y + 55), (panel_x + 10 + bar_w, panel_y + 68), bar_color, -1)

            # 12 Kare Dogrulandiginda: Hemen yazma, 5 saniye (150 kare) sonrasini toplamaya basla!
            if is_confirmed:
                if not in_cooldown and not recording_post_event:
                    recording_post_event = True
                    post_event_counter = 0
                    event_timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    print(f"\n[+] SIGARA DOGRULANDI! 5 saniyelik sonrasi kaydediliyor ({POST_EVENT_FRAMES_REQUIRED} kare)...")

                cv2.putText(frame, f"DOGRULANDI: SIGARA (%{int(best_score*100)})", (w // 2 - 210, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Ekranda 5sn sonrasi kayit durumunu goster
            if recording_post_event:
                rec_progress = int((post_event_counter / POST_EVENT_FRAMES_REQUIRED) * 100)
                cv2.rectangle(frame, (10, h - 60), (320, h - 20), (0, 0, 180), -1)
                cv2.putText(frame, f"KAYDEDILIYOR (+5sn): %{rec_progress}", (20, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # Ekranda en son kaydedilen videoyu goster
            if last_saved_video_name and not recording_post_event:
                cv2.putText(frame, f"Son Video: {last_saved_video_name}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)

            # Sol ust Durum Bilgisi
            if in_cooldown:
                mins = int(cooldown_remaining // 60)
                secs = int(cooldown_remaining % 60)
                cd_str = f" | 15dk Bekleme: {mins}d {secs:02d}s"
            elif recording_post_event:
                cd_str = f" | Video Hazirlaniyor ({post_event_counter}/150)"
            else:
                cd_str = " | Bildirim/Video: Hazir"

            status_text = f"FPS: {fps} | " + ("ENGEL/KAGIT!" if is_camera_obstructed else ("ONAYLANDI!" if is_confirmed else "NORMAL")) + cd_str
            status_color = (0, 0, 255) if (is_confirmed or is_camera_obstructed) else (0, 255, 0)
            cv2.putText(frame, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_color, 1)

            cv2.imshow("Sigara Takip Sistemi (5sn Once + 5sn Sonra Video)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
