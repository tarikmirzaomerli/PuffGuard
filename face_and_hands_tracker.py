import cv2
import mediapipe as mp
import math
import time
import os
import threading
import numpy as np
import pyttsx3
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

def speak_text_async(text):
    """Turkce sesli ikazi arka planda ayri bir thread icinde calistirir (Kamerayi dondurmez)."""
    def _speak():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150) # Dogal konusma hizi

            # Turkce ses motoru secimi
            voices = engine.getProperty('voices')
            for voice in voices:
                if 'turkish' in voice.name.lower() or 'tr' in voice.id.lower():
                    engine.setProperty('voice', voice.id)
                    break

            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[!] Sesli ikaz hatasi: {e}")

    threading.Thread(target=_speak, daemon=True).start()

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

def save_video_async(frames_list, video_path, width, height, fps=30.0, event_name="VIDEO"):
    """Tamponu arka planda MP4 olarak kaydeder (Kamera akisini asla dondurmez)."""
    def _save():
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            for f in frames_list:
                out.write(f)
            out.release()
            print(f"\n[+] {event_name} KAYDEDILDI -> {video_path}")
        except Exception as e:
            print(f"[!] Video kaydetme hatasi: {e}")

    threading.Thread(target=_save, daemon=True).start()

def calculate_distance(p1, p2):
    """Iki 2D nokta arasindaki Oklid mesafesini hesaplar."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_ear(eye_points, landmarks, width, height):
    """MediaPipe nirengi noktalarindan Eye Aspect Ratio (EAR) hesaplar."""
    coords = []
    for idx in eye_points:
        lm = landmarks[idx]
        coords.append((lm.x * width, lm.y * height))

    v1 = calculate_distance(coords[1], coords[5]) # P2 - P6
    v2 = calculate_distance(coords[2], coords[4]) # P3 - P5
    h = calculate_distance(coords[0], coords[3])  # P1 - P4

    if h == 0:
        return 0.3

    ear = (v1 + v2) / (2.0 * h)
    return ear

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

    # --- PARAMETRELER (SIGARA) ---
    CONFIDENCE_THRESHOLD = 0.65                # Guven Esigi (%65)
    REQUIRED_CONSECUTIVE_FRAMES = 12           # 12 Kesintisiz Kare Dogrulamasi
    ALLOWED_CLASSES = {"cigarette"}            # Hedef sinif
    CIGARETTE_NOTIFICATION_COOLDOWN = 900.0    # 15 DAKIKA (900 saniye) Cooldown

    # --- PARAMETRELER (UYKU / DROWSINESS - 1 DAKIKA KURALI) ---
    EAR_THRESHOLD = 0.20                       # EAR < 0.20 ise goz kapali
    SLEEP_DURATION_REQ_SEC = 60.0              # Kesintisiz 1 DAKIKA (60 saniye)
    SLEEP_NOTIFICATION_COOLDOWN = 120.0        # 2 DAKIKA (120 saniye) Uyku Cooldown

    # MediaPipe Goz Landmark Indeksleri
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

    # --- PARAMETRELER (KAMERA ENGELLEME - CANNY KENAR & DOKU ANALIZI) ---
    BLOCK_DURATION_REQ_SEC = 60.0              # Kesintisiz 1 DAKIKA (60 saniye)
    SECURITY_NOTIFICATION_COOLDOWN = 120.0     # 2 DAKIKA (120 saniye) Cooldown
    BRIGHTNESS_OBSTRUCTION_THRESHOLD = 22.0    # Parlaklik < 22 ise karartma/el
    EDGE_COUNT_THRESHOLD = 150                 # Canny kenar sayisi < 150 ise lense yapisik engel

    # 10 SANIYELIK (300 KARE) SUREKLI DONEN TAMPON
    frame_buffer = deque(maxlen=300)

    # Sigara 5sn Oncesi + 5sn Sonrasi Kayit Degiskenleri
    POST_EVENT_FRAMES_REQUIRED = 150
    recording_post_event = False
    post_event_counter = 0
    event_timestamp_str = ""

    # Zamanlayici ve durum degiskenleri
    consecutive_detection_count = 0
    sleep_start_time = None
    block_start_time = None
    camera_loss_start_time = None
    last_cigarette_notification_time = 0.0
    last_sleep_notification_time = 0.0
    last_camera_loss_time = 0.0
    last_blocked_notification_time = 0.0
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
            img_dim = max(160, (roi_img.shape[0] // 32) * 32)
            results = yolo_model(roi_img, conf=CONFIDENCE_THRESHOLD, verbose=False, imgsz=img_dim)
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
        print("PuffGuard - Sesli Ikazli Guvenlik & Takip Sistemi Aktif.")
        print(f"- Ses Motoru: pyttsx3 (Turkce, 150 WPM)")
        print(f"- Kamera Engelleme: Kağıt, Bez, Bant, Karartma (Hassas Canny Algılama)")
        print(f"- Masadan Ayrılma: Boş oda kenarları algılanır, sessizce beklenir.")
        print(f"- Uyku Tespiti: Kesintisiz {int(SLEEP_DURATION_REQ_SEC)}s (1 Dakika)")
        print(f"- Sigara Cooldown: {int(CIGARETTE_NOTIFICATION_COOLDOWN)}s (15 Dakika)")
        print("- Cikis: 'q' tusu")
        print("==================================================")

        while True:
            current_time = time.time()
            success, frame = cap.read()

            # --- 1. DONANIM / YAZILIM KAPANMASI (FRAME LOSS) ---
            if not success or frame is None:
                if camera_loss_start_time is None:
                    camera_loss_start_time = current_time
                loss_duration = current_time - camera_loss_start_time

                if loss_duration >= BLOCK_DURATION_REQ_SEC:
                    if (current_time - last_camera_loss_time) >= SECURITY_NOTIFICATION_COOLDOWN:
                        print("\n[!] UYARI: Kamera 1 Dakikadır Bağlantısız veya Kapatıldı!")
                        if last_valid_frame is not None:
                            now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                            closed_photo_name = f"camera_closed_{now_str}.jpg"
                            closed_photo_path = os.path.join(photo_dir, closed_photo_name)
                            cv2.imwrite(closed_photo_path, last_valid_frame)
                            print(f"[+] Kapanmadan onceki kare kaydedildi: {closed_photo_path}")

                        send_desktop_notification(
                            "UYARI: Kamera 1 Dakikadır Engellendi!",
                            "Kamera görüntüsü 1 dakikadır alınamıyor. Son geçerli durum kaydedildi."
                        )
                        speak_text_async("Uyarı! Kamera görüşü engellendi.")
                        last_camera_loss_time = current_time
                time.sleep(0.1)
                continue
            else:
                camera_loss_start_time = None

            last_valid_frame = frame.copy()
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # 300 KARELIK TAMPONA SUREKLI EKLE
            frame_buffer.append(frame.copy())

            # --- SIGARA ICIN 5 SANIYE SONRASI KARE TOPLAMA VE KAYIT ---
            if recording_post_event:
                post_event_counter += 1
                if post_event_counter >= POST_EVENT_FRAMES_REQUIRED:
                    video_name = f"cigarette_video_10s_{event_timestamp_str}.mp4"
                    video_path = os.path.join(photo_dir, video_name)

                    save_video_async(list(frame_buffer), video_path, w, h, fps=30.0, event_name="10s SIGARA VIDEOSU")
                    last_saved_video_name = video_name

                    send_desktop_notification(
                        "Uyarı: Sigara Doğrulandı!",
                        f"5 sn öncesi ve 5 sn sonrasını içeren 10 saniyelik video '{video_name}' kaydedildi."
                    )

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

            has_face = bool(face_results.multi_face_landmarks)

            # --- 2. HASSAS CANNY KENAR & DOKU ANALIZI ILE ENGELLEME TESPITI ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avg_brightness = float(np.mean(gray))
            std_brightness = float(np.std(gray))
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            edges = cv2.Canny(gray, 40, 120)
            edge_count = int(np.count_nonzero(edges))

            is_camera_obstructed = False
            block_reason = ""

            if has_face:
                is_camera_obstructed = False
            else:
                if avg_brightness < BRIGHTNESS_OBSTRUCTION_THRESHOLD:
                    is_camera_obstructed = True
                    block_reason = "Karartma / Siyah Kapak"
                elif edge_count < EDGE_COUNT_THRESHOLD and (std_brightness < 14.0 or laplacian_var < 25.0):
                    is_camera_obstructed = True
                    block_reason = "Kağıt / Lense Yapışık Engel"
                else:
                    is_camera_obstructed = False

            block_duration = 0.0
            if is_camera_obstructed:
                if block_start_time is None:
                    block_start_time = current_time

                block_duration = current_time - block_start_time

                # Kesintisiz 1 DAKIKA (60 saniye) engelleme durumunda bildirim ve SESLI IKAZ
                if block_duration >= BLOCK_DURATION_REQ_SEC:
                    if (current_time - last_blocked_notification_time) >= SECURITY_NOTIFICATION_COOLDOWN:
                        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        blocked_photo_name = f"camera_blocked_{now_str}.jpg"
                        blocked_photo_path = os.path.join(photo_dir, blocked_photo_name)
                        cv2.imwrite(blocked_photo_path, frame)
                        print(f"\n[!] UYARI: Kamera Önü Engellendi! Fotoğraf: {blocked_photo_name}")

                        # Masaustu Bildirimi & Turkce Sesli Ikaz
                        send_desktop_notification(
                            "UYARI: Kamera Önü Engellendi!",
                            f"Kamera görüşü 1 dakikadır kapalı/engelli ({block_reason}). Son durum fotoğrafı kaydedildi."
                        )
                        speak_text_async("Uyarı! Kamera görüşü engellendi.")

                        last_blocked_notification_time = current_time

                cv2.rectangle(frame, (20, h // 2 - 40), (w - 20, h // 2 + 40), (0, 0, 180), -1)
                cv2.putText(frame, f"UYARI: KAMERA ONU ENGELLENDI ({block_reason}) {block_duration:.1f}s / {int(BLOCK_DURATION_REQ_SEC)}s", (20, h // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
            else:
                block_start_time = None

            lip_center = None
            eye_distance = 80.0
            avg_ear = 0.35
            eyes_closed = False

            # --- 3. YUZ NIRENGI, DUDAK VE UYKU (EAR) ANALIZI ---
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_LIPS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                    )

                    # Iki Goz Arasi Mesafe
                    p_left_eye = (face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h)
                    p_right_eye = (face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h)
                    eye_distance = calculate_distance(p_left_eye, p_right_eye)

                    # Dudak Merkezi
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

                    # Eye Aspect Ratio (EAR) Hesaplama
                    left_ear = calculate_ear(LEFT_EYE_INDICES, face_landmarks.landmark, w, h)
                    right_ear = calculate_ear(RIGHT_EYE_INDICES, face_landmarks.landmark, w, h)
                    avg_ear = (left_ear + right_ear) / 2.0

                    for idx in (LEFT_EYE_INDICES + RIGHT_EYE_INDICES):
                        elm = face_landmarks.landmark[idx]
                        cv2.circle(frame, (int(elm.x * w), int(elm.y * h)), 2, (0, 255, 255), -1)

                    if avg_ear < EAR_THRESHOLD:
                        eyes_closed = True

            # --- 4. UYKU / DROWSINESS SAYACI (1 DAKIKA KURALI) ---
            sleep_cooldown_remaining = max(0.0, SLEEP_NOTIFICATION_COOLDOWN - (current_time - last_sleep_notification_time))
            in_sleep_cooldown = (current_time - last_sleep_notification_time) < SLEEP_NOTIFICATION_COOLDOWN

            sleep_duration = 0.0
            if eyes_closed and not is_camera_obstructed and has_face:
                if sleep_start_time is None:
                    sleep_start_time = current_time

                sleep_duration = current_time - sleep_start_time

                if sleep_duration >= SLEEP_DURATION_REQ_SEC:
                    cv2.rectangle(frame, (20, h - 100), (w - 20, h - 55), (0, 0, 220), -1)
                    cv2.putText(frame, "UYARI: 1 DAKIKADIR UYKU HALINDESINIZ!", (35, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

                    if not in_sleep_cooldown:
                        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        sleep_video_name = f"sleep_alert_{now_str}.mp4"
                        sleep_video_path = os.path.join(photo_dir, sleep_video_name)

                        save_video_async(list(frame_buffer), sleep_video_path, w, h, fps=30.0, event_name="1 DAKIKALIK UYKU VIDEOSU")
                        last_saved_video_name = sleep_video_name

                        # Masaustu Bildirimi & Turkce Sesli Ikaz
                        send_desktop_notification(
                            "UYARI: 1 Dakikadır Uyku Halindesiniz!",
                            f"Gözleriniz kesintisiz 1 dakikadır kapalı tespit edildi! 10 sn kanıt videosu '{sleep_video_name}' kaydedildi."
                        )
                        speak_text_async("Lütfen uyanın! Bir dakikadır uyku halindesiniz.")

                        last_sleep_notification_time = current_time
                        print(f"\n[!] 1 DAKIKALIK UYKU ALARMI: 2 dakikalık (120s) uyku soğuma süresi başlatıldı.")
            else:
                sleep_start_time = None

            # 5. El Landmarklari
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
                    cv2.circle(frame, (ix, iy), 7, (0, 255, 0), cv2.FILLED)

            # --- 6. 2.2x CARPANLI VE MINIMUM 150px SINIRLI DINAMIK AGIZ ROI ---
            if lip_center is not None and not is_camera_obstructed:
                calculated_size = int(eye_distance * 2.2)
                roi_size = max(150, calculated_size)
                half_sz = roi_size // 2

                rx1 = max(0, lip_center[0] - half_sz)
                ry1 = max(0, lip_center[1] - half_sz)
                rx2 = min(w, rx1 + roi_size)
                ry2 = min(h, ry1 + roi_size)

                if rx2 - rx1 < roi_size and rx1 > 0:
                    rx1 = max(0, rx2 - roi_size)
                if ry2 - ry1 < roi_size and ry1 > 0:
                    ry1 = max(0, ry2 - roi_size)

                mouth_crop = frame[ry1:ry2, rx1:rx2]

                if yolo_model is not None and mouth_crop.size > 0 and not is_yolo_busy:
                    is_yolo_busy = True
                    threading.Thread(target=run_yolo_on_mouth_roi, args=(mouth_crop.copy(), rx1, ry1), daemon=True).start()

                has_cig = len(detected_objects) > 0
                roi_box_color = (0, 0, 255) if has_cig else (255, 255, 0)
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), roi_box_color, 2)
                cv2.putText(frame, f"Dinamik ROI {roi_size}x{roi_size} (2.2x)", (rx1, max(15, ry1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, roi_box_color, 1)

            # 7. Sigara Dogrulama Sayaci (12 Kare)
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

            # 8. Sigara 15 Dk Cooldown ve Tetikleme
            cig_cooldown_remaining = max(0.0, CIGARETTE_NOTIFICATION_COOLDOWN - (current_time - last_cigarette_notification_time))
            in_cig_cooldown = (current_time - last_cigarette_notification_time) < CIGARETTE_NOTIFICATION_COOLDOWN
            is_cig_confirmed = consecutive_detection_count >= REQUIRED_CONSECUTIVE_FRAMES

            # 12 Kare Sigara Dogrulandiginda (Bildirim, Sesli Ikaz ve 5sn Sonrasi Video Kaydi)
            if is_cig_confirmed:
                if not in_cig_cooldown and not recording_post_event:
                    recording_post_event = True
                    post_event_counter = 0
                    event_timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    print(f"\n[+] SIGARA DOGRULANDI! 5 saniyelik sonrasi kaydediliyor...")

                    # Turkce Sesli Ikaz
                    speak_text_async("Sigara kullanımı tespit edildi. Kayıt alınıyor.")

                cv2.putText(frame, f"DOGRULANDI: SIGARA (%{int(best_score*100)})", (w // 2 - 210, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Sag ust Bilgi Paneli
            panel_w, panel_h = 320, 115
            panel_x = w - panel_w - 10
            panel_y = 10

            overlay = frame.copy()
            bg_col = (0, 0, 200) if (is_cig_confirmed or sleep_duration >= SLEEP_DURATION_REQ_SEC) else (40, 40, 40)
            cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), bg_col, -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
            cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (255, 255, 255), 1)

            cv2.putText(frame, "PUFFGUARD TAKIP PANELI", (panel_x + 10, panel_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
            
            # Sigara Sayac Satiri
            counter_str = f"Sigara: {consecutive_detection_count}/{REQUIRED_CONSECUTIVE_FRAMES}" + (" (15dk Bekleme)" if in_cig_cooldown else " (Hazir)")
            cv2.putText(frame, counter_str, (panel_x + 10, panel_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)

            # Uyku & EAR Satiri
            ear_status_color = (0, 0, 255) if eyes_closed else (0, 255, 0)
            ear_str = f"Goz (EAR): {avg_ear:.2f} | Kapali: {sleep_duration:.1f}s / {int(SLEEP_DURATION_REQ_SEC)}s"
            cv2.putText(frame, ear_str, (panel_x + 10, panel_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.40, ear_status_color, 1)

            # Uyku Ilerleme Cubugu
            sleep_prog = min(1.0, sleep_duration / SLEEP_DURATION_REQ_SEC)
            s_bar_w = int((panel_w - 20) * sleep_prog)
            cv2.rectangle(frame, (panel_x + 10, panel_y + 75), (panel_x + panel_w - 10, panel_y + 86), (60, 60, 60), -1)
            cv2.rectangle(frame, (panel_x + 10, panel_y + 75), (panel_x + 10 + s_bar_w, panel_y + 86), (0, 165, 255), -1)

            # Durum / Cooldown Bilgisi
            sleep_cd_text = f"Uyku CD: {int(sleep_cooldown_remaining)}s" if in_sleep_cooldown else "Uyku Modu: Aktif (1dk)"
            cv2.putText(frame, sleep_cd_text, (panel_x + 10, panel_y + 103), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

            # Ekranda 5sn sonrasi kayit durumunu goster
            if recording_post_event:
                rec_progress = int((post_event_counter / POST_EVENT_FRAMES_REQUIRED) * 100)
                cv2.rectangle(frame, (10, h - 60), (320, h - 20), (0, 0, 180), -1)
                cv2.putText(frame, f"KAYDEDILIYOR (+5sn): %{rec_progress}", (20, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # Ekranda en son kaydedilen videoyu goster
            if last_saved_video_name and not recording_post_event:
                cv2.putText(frame, f"Son Video: {last_saved_video_name}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)

            # Sol ust Durum Bilgisi
            if in_cig_cooldown:
                mins = int(cig_cooldown_remaining // 60)
                secs = int(cig_cooldown_remaining % 60)
                cd_str = f" | Sigara 15dk CD: {mins}d {secs:02d}s"
            elif recording_post_event:
                cd_str = f" | Video Hazirlaniyor ({post_event_counter}/150)"
            else:
                cd_str = " | Sigara: Hazir"

            # Durum Etiketi
            if is_camera_obstructed:
                system_status_tag = "KAMERA ENGELLENDI!"
                system_status_color = (0, 0, 255)
            elif not has_face:
                system_status_tag = "MASADAN AYRILDI (BEKLEMEDE)"
                system_status_color = (255, 200, 0)
            elif sleep_duration >= SLEEP_DURATION_REQ_SEC:
                system_status_tag = "1DK UYKU ALARMI!"
                system_status_color = (0, 0, 255)
            elif is_cig_confirmed:
                system_status_tag = "SIGARA ONAYLANDI!"
                system_status_color = (0, 0, 255)
            else:
                system_status_tag = "NORMAL (AKTIF)"
                system_status_color = (0, 255, 0)

            status_text = f"FPS: {fps} | {system_status_tag}{cd_str}"
            cv2.putText(frame, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.40, system_status_color, 1)

            cv2.imshow("PuffGuard - Sigara & Uyku Takip Sistemi", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
