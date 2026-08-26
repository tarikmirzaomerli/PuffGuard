import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
torch.set_num_threads(1)

import cv2
import mediapipe as mp
import math
import time
import gc
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

# Proje Kok Dizini
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

def speak_text_async(text):
    """Windows OneCore Turkce (Tolga) dogal ses motoru ile anlasilir sesli ikaz verir."""
    def _speak():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)

            onecore_turkish_voices = [
                r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech_OneCore\Voices\Tokens\MSTTS_V110_trTR_Tolga",
                r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech_OneCore\Voices\Tokens\MSTTS_V110_trTR_Emel"
            ]

            voice_selected = False
            for v_id in onecore_turkish_voices:
                try:
                    engine.setProperty('voice', v_id)
                    voice_selected = True
                    break
                except Exception:
                    pass

            if not voice_selected:
                voices = engine.getProperty('voices')
                for voice in voices:
                    if 'tr' in voice.id.lower() or 'turkish' in voice.name.lower():
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

def save_compressed_video_async(compressed_frames_list, video_path, width=640, height=360, fps=30.0, event_name="VIDEO"):
    """Sikistirilmis JPEG tamponunu decode edip MP4 yazar ve RAM'i tamamen serbest birakir."""
    def _save():
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            for enc_frame in compressed_frames_list:
                raw_frame = cv2.imdecode(enc_frame, cv2.IMREAD_COLOR)
                if raw_frame is not None:
                    out.write(raw_frame)
            out.release()
            print(f"\n[+] {event_name} KAYDEDILDI -> {video_path}")
        except Exception as e:
            print(f"[!] Video kaydetme hatasi: {e}")
        finally:
            del compressed_frames_list
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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

def load_lightweight_onnx_model():
    """YOLO modelini 640x640 ONNX Runtime formatiyla yukler."""
    print("==================================================")
    print("[+] 640x640 ONNX Modeli kontrol ediliyor...")
    onnx_path = os.path.join(PROJECT_DIR, "best.onnx")
    pt_path = os.path.join(PROJECT_DIR, "best.pt")

    if not os.path.exists(onnx_path) and os.path.exists(pt_path):
        print("[+] 'best.pt' ONNX formatina donusturuluyor (imgsz=640)...")
        temp_model = YOLO(pt_path)
        temp_model.export(format="onnx", imgsz=640)
        del temp_model
        gc.collect()

    if os.path.exists(onnx_path):
        print(f"[+] 640x640 ONNX Runtime Modeli yukleniyor: '{onnx_path}'")
        model = YOLO(onnx_path, task='detect')
        print(f"[+] ONNX Modeli basariyla yuklendi! Siniflar: {model.names}")
        print("==================================================")
        return model
    elif os.path.exists(pt_path):
        print(f"[!] ONNX bulunamadi, 'best.pt' yukleniyor...")
        return YOLO(pt_path)
    else:
        print("[!] Model dosyasi bulunamadi!")
        return None

def main():
    # 1. Proje Icindeki 'cigarettes-foto' Klasorunu Olustur
    photo_dir = os.path.join(PROJECT_DIR, 'cigarettes-foto')
    if not os.path.exists(photo_dir):
        os.makedirs(photo_dir, exist_ok=True)
        print(f"[+] Kayit klasoru olusturuldu: {photo_dir}")
    else:
        print(f"[+] Kayit klasoru hazir: {photo_dir}")

    # 2. 640x640 ONNX Modelini Yukleme
    yolo_model = load_lightweight_onnx_model()

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

    # Kamera cozunurlugunu 640x480 olarak sabitle
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # --- PARAMETRELER (SIGARA - HASSAS ONNX AYARLARI) ---
    CONFIDENCE_THRESHOLD = 0.50                # %50 Guven Esigi (Hassas ONNX tespiti)
    REQUIRED_CONSECUTIVE_FRAMES = 12           # 12 Kesintisiz Kare Dogrulamasi
    ALLOWED_CLASSES = {"cigarette"}            # Hedef sinif
    CIGARETTE_NOTIFICATION_COOLDOWN = 900.0    # 15 DAKIKA (900 saniye) Cooldown
    YOLO_BUFFER_WINDOW_SEC = 3.0               # El uzaklassa bile 3 saniye tetikte kal

    # --- PARAMETRELER (UYKU / DROWSINESS - 1 DAKIKA KURALI) ---
    EAR_THRESHOLD = 0.20                       # EAR < 0.20 ise goz kapali
    SLEEP_DURATION_REQ_SEC = 60.0              # Kesintisiz 1 DAKIKA (60 saniye)
    SLEEP_NOTIFICATION_COOLDOWN = 120.0        # 2 DAKIKA (120 saniye) Uyku Cooldown

    # MediaPipe Goz Landmark Indeksleri
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

    # --- PARAMETRELER (KAMERA ENGELLEME) ---
    BLOCK_DURATION_REQ_SEC = 60.0              # Kesintisiz 1 DAKIKA (60 saniye)
    SECURITY_NOTIFICATION_COOLDOWN = 120.0     # 2 DAKIKA (120 saniye) Cooldown
    BRIGHTNESS_OBSTRUCTION_THRESHOLD = 22.0    # Parlaklik < 22 ise karartma/el
    EDGE_COUNT_THRESHOLD = 150                 # Canny kenar sayisi < 150 ise lense yapisik engel

    # JPEG SIKISTIRILMIS 300 KARELIK TAMPON (RAM SADECE ~25 MB)
    BUFFER_W, BUFFER_H = 640, 360
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
    last_hand_near_mouth_time = 0.0
    last_cigarette_notification_time = 0.0
    last_sleep_notification_time = 0.0
    last_camera_loss_time = 0.0
    last_blocked_notification_time = 0.0
    last_valid_frame = None
    last_saved_video_name = ""

    frame_counter = 0
    fps_start_time = time.time()
    fps = 0

    # Thread / Asenkron degiskenler
    is_yolo_busy = False
    detected_objects = []
    last_seen_time = 0.0

    # Frame Skipping onbellek degiskenleri
    cached_has_face = False
    cached_lip_center = None
    cached_eye_distance = 80.0
    cached_avg_ear = 0.35
    cached_eyes_closed = False
    cached_is_obstructed = False
    cached_block_reason = ""
    cached_hand_pos = None

    lip_indices = set()
    for conn in mp_face_mesh.FACEMESH_LIPS:
        lip_indices.add(conn[0])
        lip_indices.add(conn[1])

    def run_yolo_on_mouth_roi(roi_img, orig_w, orig_h, offset_x, offset_y):
        nonlocal is_yolo_busy, detected_objects, last_seen_time
        try:
            # 640x640 Yeniden Boyutlandırma ile Yuksek ONNX Hassasiyeti
            resized_roi = cv2.resize(roi_img, (640, 640))
            scale_x = orig_w / 640.0
            scale_y = orig_h / 640.0

            with torch.no_grad():
                results = yolo_model(resized_roi, conf=CONFIDENCE_THRESHOLD, verbose=False, imgsz=640)
                current_boxes = []

                for r in results:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = r.names[cls_id].strip().lower() if (hasattr(r, 'names') and cls_id in r.names) else "cigarette"

                        if (cls_name in ALLOWED_CLASSES or cls_id == 0) and conf >= CONFIDENCE_THRESHOLD:
                            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                            # 640x640 kutusunu orijinal kirpma koordinatina olcekle
                            fx1 = offset_x + int(bx1 * scale_x)
                            fy1 = offset_y + int(by1 * scale_y)
                            fx2 = offset_x + int(bx2 * scale_x)
                            fy2 = offset_y + int(by2 * scale_y)
                            current_boxes.append((fx1, fy1, fx2, fy2, cls_name, conf))
                            last_seen_time = time.time()
                            print(f"[*] SIGARA YAKALANDI (ONNX): %{int(conf*100)}")

                detected_objects = current_boxes
                del results, resized_roi
        except Exception as e:
            pass
        finally:
            del roi_img
            is_yolo_busy = False

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh, mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        print("==================================================")
        print("PuffGuard - 640x640 ONNX & 3s Buffer Trigger Aktif.")
        print(f"- Model: ONNX Runtime 640x640 (conf={int(CONFIDENCE_THRESHOLD*100)}%)")
        print(f"- Genişletilmiş Tetikleme: El ağza geldikten sonra {int(YOLO_BUFFER_WINDOW_SEC)}s aktif kalır.")
        print(f"- Dairesel Tampon: 300 Kare JPEG (~25 MB RAM)")
        print(f"- Ses Motoru: Microsoft Tolga (Doğal Türkçe)")
        print(f"- Uyku Tespiti: Kesintisiz {int(SLEEP_DURATION_REQ_SEC)}s (1 Dakika)")
        print(f"- Sigara Cooldown: {int(CIGARETTE_NOTIFICATION_COOLDOWN)}s (15 Dakika)")
        print("- Çıkış: 'q' tuşu")
        print("==================================================")

        while True:
            current_time = time.time()
            frame_counter += 1
            process_ai_this_frame = (frame_counter % 2 == 0)

            success, frame = cap.read()

            # --- 1. DONANIM / YAZILIM KAPANMASI ---
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

            last_valid_frame = frame
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Sıkıştırılmış JPEG Tampon
            small_resized = cv2.resize(frame, (BUFFER_W, BUFFER_H))
            _, enc_frame = cv2.imencode('.jpg', small_resized, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_buffer.append(enc_frame)
            del small_resized

            # --- SIGARA ICIN 5 SANIYE SONRASI KARE TOPLAMA VE KAYIT ---
            if recording_post_event:
                post_event_counter += 1
                if post_event_counter >= POST_EVENT_FRAMES_REQUIRED:
                    video_name = f"cigarette_video_10s_{event_timestamp_str}.mp4"
                    video_path = os.path.join(photo_dir, video_name)

                    save_compressed_video_async(list(frame_buffer), video_path, width=BUFFER_W, height=BUFFER_H, fps=30.0, event_name="10s SIGARA VIDEOSU")
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
            if time.time() - fps_start_time >= 1.0:
                fps = frame_counter
                frame_counter = 0
                fps_start_time = time.time()
                gc.collect()

            # --- FRAME SKIPPING: 2 KAREDE 1 AI ANALIZI ---
            if process_ai_this_frame:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False

                face_results = face_mesh.process(rgb_frame)
                hand_results = hands.process(rgb_frame)

                rgb_frame.flags.writeable = True

                cached_has_face = bool(face_results.multi_face_landmarks)

                # Engel / Karartma / Doku Analizi
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                avg_brightness = float(np.mean(gray))
                std_brightness = float(np.std(gray))
                laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

                edges = cv2.Canny(gray, 40, 120)
                edge_count = int(np.count_nonzero(edges))

                if cached_has_face:
                    cached_is_obstructed = False
                    cached_block_reason = ""
                else:
                    if avg_brightness < BRIGHTNESS_OBSTRUCTION_THRESHOLD:
                        cached_is_obstructed = True
                        cached_block_reason = "Karartma / Siyah Kapak"
                    elif edge_count < EDGE_COUNT_THRESHOLD and (std_brightness < 14.0 or laplacian_var < 25.0):
                        cached_is_obstructed = True
                        cached_block_reason = "Kağıt / Lense Yapışık Engel"
                    else:
                        cached_is_obstructed = False
                        cached_block_reason = ""

                del gray, edges

                cached_lip_center = None
                cached_eyes_closed = False

                if face_results.multi_face_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        p_left_eye = (face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h)
                        p_right_eye = (face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h)
                        cached_eye_distance = calculate_distance(p_left_eye, p_right_eye)

                        lip_x = []
                        lip_y = []
                        for idx in lip_indices:
                            lm = face_landmarks.landmark[idx]
                            lip_x.append(int(lm.x * w))
                            lip_y.append(int(lm.y * h))

                        if lip_x and lip_y:
                            cached_lip_center = (int(sum(lip_x) / len(lip_x)), int(sum(lip_y) / len(lip_y)))

                        left_ear = calculate_ear(LEFT_EYE_INDICES, face_landmarks.landmark, w, h)
                        right_ear = calculate_ear(RIGHT_EYE_INDICES, face_landmarks.landmark, w, h)
                        cached_avg_ear = (left_ear + right_ear) / 2.0

                        if cached_avg_ear < EAR_THRESHOLD:
                            cached_eyes_closed = True

                # El Konumu Takibi
                cached_hand_pos = None
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                        cached_hand_pos = (ix, iy)
                        cv2.circle(frame, (ix, iy), 6, (0, 255, 0), cv2.FILLED)

                del face_results, hand_results, rgb_frame

            has_face = cached_has_face
            is_camera_obstructed = cached_is_obstructed
            block_reason = cached_block_reason
            lip_center = cached_lip_center
            eye_distance = cached_eye_distance
            avg_ear = cached_avg_ear
            eyes_closed = cached_eyes_closed
            hand_pos = cached_hand_pos

            # --- 2. ENGELLEME SURE VE IKAZ SAYACI ---
            block_duration = 0.0
            if is_camera_obstructed:
                if block_start_time is None:
                    block_start_time = current_time

                block_duration = current_time - block_start_time

                if block_duration >= BLOCK_DURATION_REQ_SEC:
                    if (current_time - last_blocked_notification_time) >= SECURITY_NOTIFICATION_COOLDOWN:
                        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        blocked_photo_name = f"camera_blocked_{now_str}.jpg"
                        blocked_photo_path = os.path.join(photo_dir, blocked_photo_name)
                        cv2.imwrite(blocked_photo_path, frame)
                        print(f"\n[!] UYARI: Kamera Önü Engellendi! Fotoğraf: {blocked_photo_name}")

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

            # --- 3. UYKU SAYACI ---
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

                        save_compressed_video_async(list(frame_buffer), sleep_video_path, width=BUFFER_W, height=BUFFER_H, fps=30.0, event_name="1 DAKIKALIK UYKU VIDEOSU")
                        last_saved_video_name = sleep_video_name

                        send_desktop_notification(
                            "UYARI: 1 Dakikadır Uyku Halindesiniz!",
                            f"Gözleriniz kesintisiz 1 dakikadır kapalı tespit edildi! 10 sn kanıt videosu '{sleep_video_name}' kaydedildi."
                        )
                        speak_text_async("Lütfen uyanın! Bir dakikadır uyku halindesiniz.")
                        last_sleep_notification_time = current_time
                        print(f"\n[!] 1 DAKIKALIK UYKU ALARMI: 2 dakikalık (120s) uyku soğuma süresi başlatıldı.")
            else:
                sleep_start_time = None

            # --- 4. GENISLETILMIS TETIKLEME (3sn BUFFER) & 640x640 ONNX YOLO ---
            is_yolo_active = False
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

                # 1. El ağza yakın mı kontrolü
                if hand_pos is not None:
                    hand_dist = calculate_distance(hand_pos, lip_center)
                    if hand_dist <= (roi_size * 1.25):
                        last_hand_near_mouth_time = current_time

                # 2. Genişletilmiş Tetikleme Penceresi: El uzaklaşsa bile 3 saniye boyunca YOLO aktif kalır!
                if (current_time - last_hand_near_mouth_time) <= YOLO_BUFFER_WINDOW_SEC:
                    is_yolo_active = True

                mouth_crop = frame[ry1:ry2, rx1:rx2]
                cw = rx2 - rx1
                ch = ry2 - ry1

                # YOLO YALNIZCA TETIKLEME PENCERESINDEYKEN VE AI KARESIYSE CALISIR!
                if yolo_model is not None and mouth_crop.size > 0 and not is_yolo_busy and process_ai_this_frame and is_yolo_active:
                    is_yolo_busy = True
                    threading.Thread(target=run_yolo_on_mouth_roi, args=(mouth_crop.copy(), cw, ch, rx1, ry1), daemon=True).start()
                elif not is_yolo_active:
                    if (current_time - last_seen_time) > 0.5:
                        detected_objects = []

                has_cig = len(detected_objects) > 0
                roi_box_color = (0, 0, 255) if has_cig else ((0, 255, 255) if is_yolo_active else (120, 120, 120))
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), roi_box_color, 2)
                
                remaining_buf = max(0.0, YOLO_BUFFER_WINDOW_SEC - (current_time - last_hand_near_mouth_time))
                roi_label = f"ONNX ROI {roi_size}x{roi_size}" + (f" [YOLO AKTIF {remaining_buf:.1f}s]" if is_yolo_active else " [BEKLEMEDE]")
                cv2.putText(frame, roi_label, (rx1, max(15, ry1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, roi_box_color, 1)

            # 5. Sigara Dogrulama Sayaci (12 Kare)
            is_currently_detected = len(detected_objects) > 0 or (current_time - last_seen_time < 0.45)
            best_score = 0.0

            if is_currently_detected and not is_camera_obstructed and is_yolo_active:
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

            # 6. Sigara 15 Dk Cooldown ve Tetikleme
            cig_cooldown_remaining = max(0.0, CIGARETTE_NOTIFICATION_COOLDOWN - (current_time - last_cigarette_notification_time))
            in_cig_cooldown = (current_time - last_cigarette_notification_time) < CIGARETTE_NOTIFICATION_COOLDOWN
            is_cig_confirmed = consecutive_detection_count >= REQUIRED_CONSECUTIVE_FRAMES

            if is_cig_confirmed:
                if not in_cig_cooldown and not recording_post_event:
                    recording_post_event = True
                    post_event_counter = 0
                    event_timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    print(f"\n[+] SIGARA DOGRULANDI! 5 saniyelik sonrasi kaydediliyor...")
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
            
            counter_str = f"Sigara: {consecutive_detection_count}/{REQUIRED_CONSECUTIVE_FRAMES}" + (" (15dk Bekleme)" if in_cig_cooldown else " (Hazir)")
            cv2.putText(frame, counter_str, (panel_x + 10, panel_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)

            ear_status_color = (0, 0, 255) if eyes_closed else (0, 255, 0)
            ear_str = f"Goz (EAR): {avg_ear:.2f} | Kapali: {sleep_duration:.1f}s / {int(SLEEP_DURATION_REQ_SEC)}s"
            cv2.putText(frame, ear_str, (panel_x + 10, panel_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.40, ear_status_color, 1)

            sleep_prog = min(1.0, sleep_duration / SLEEP_DURATION_REQ_SEC)
            s_bar_w = int((panel_w - 20) * sleep_prog)
            cv2.rectangle(frame, (panel_x + 10, panel_y + 75), (panel_x + panel_w - 10, panel_y + 86), (60, 60, 60), -1)
            cv2.rectangle(frame, (panel_x + 10, panel_y + 75), (panel_x + 10 + s_bar_w, panel_y + 86), (0, 165, 255), -1)

            sleep_cd_text = f"Uyku CD: {int(sleep_cooldown_remaining)}s" if in_sleep_cooldown else "Uyku Modu: Aktif (1dk)"
            cv2.putText(frame, sleep_cd_text, (panel_x + 10, panel_y + 103), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

            if recording_post_event:
                rec_progress = int((post_event_counter / POST_EVENT_FRAMES_REQUIRED) * 100)
                cv2.rectangle(frame, (10, h - 60), (320, h - 20), (0, 0, 180), -1)
                cv2.putText(frame, f"KAYDEDILIYOR (+5sn): %{rec_progress}", (20, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            if last_saved_video_name and not recording_post_event:
                cv2.putText(frame, f"Son Video: {last_saved_video_name}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)

            if in_cig_cooldown:
                mins = int(cig_cooldown_remaining // 60)
                secs = int(cig_cooldown_remaining % 60)
                cd_str = f" | Sigara 15dk CD: {mins}d {secs:02d}s"
            elif recording_post_event:
                cd_str = f" | Video Hazirlaniyor ({post_event_counter}/150)"
            else:
                cd_str = " | Sigara: Hazir"

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
    gc.collect()

if __name__ == "__main__":
    main()
