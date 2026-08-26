# PuffGuard - AI-Powered Smoking & Drowsiness Detection System 🚬😴🛡️

Laptop kamerası üzerinden gerçek zamanlı **sigara tüketimi**, **el-ağız hareketi**, **uyku/uyuşukluk (drowsiness)** ve **kamera güvenlik durumunu** takip eden; masaüstü bildirimi gönderip tespit anında otomatik video kaydı alan çok amaçlı hibrit yapay zeka sistemi.

---

## ✨ Özellikler

### 1. 🚬 Sigara Tespiti (İki Aşamalı Hibrit Model)
- **1. Aşama:** MediaPipe Face Mesh ile hafif ve CPU dostu dudak ve el takibi.
- **2. Aşama:** Ağız bölgesi etrafında çalışan yerel YOLOv8 ile yüksek doğruluklu sigara tespiti (`conf >= %65`).
- **Doğrulama Sayacı:** Yanlış alarmları önlemek için 12 karelik kesintisiz doğrulama mantığı.
- **Gecikmeli Video Kaydı:** Sigara tespit edildiğinde olayın **5 saniye öncesini ve 5 saniye sonrasını** kapsayan toplam 10 saniyelik MP4 video kaydı (`cigarette_video_10s_YYYYMMDD_HHMMSS.mp4`).
- **15 Dakikalık (900 sn) Cooldown:** Üst üste bildirim yağmurunu önleyen bağımsız soğuma süresi.

### 2. 😴 Uyku & Uyuşukluk Tespiti (Eye Aspect Ratio - EAR)
- MediaPipe yüz nirengi noktaları üzerinden sağ ve sol gözün dikey ve yatay mesafelerini sürekli analiz ederek **EAR (Eye Aspect Ratio)** değerini hesaplar.
- **Göz Kapalı Eşiği:** `EAR < 0.20` olduğunda gözler kapalı kabul edilir.
- **3 Saniyelik (90 Kare) Kesintisiz Alarm:** Gözler aralıksız 3 saniye kapalı kaldığında **"UYARI: Uykuya Daldınız!"** masaüstü uyarısı tetiklenir.
- **Kanıt Videosu:** Olay anının 10 saniyelik videosu `sleep_alert_YYYYMMDD_HHMMSS.mp4` adıyla kaydedilir.
- **2 Dakikalık (120 sn) Cooldown:** Uyku alarmları için bağımsız 2 dakikalık soğuma süresi.

### 3. 🛡️ Kamera Güvenlik ve Engel Kontrolü
- Kameranın kapatılması, kağıt/bezle engellenmesi veya karartılması durumunda doku ve parlaklık analizi ile otomatik uyarı verir ve kanıt fotoğrafı kaydeder.

### 4. ⚡ %100 Ücretsiz & Yerel
- Tüm işlemler yerel CPU/GPU üzerinde çalışır, harici bulut API'si veya ücretli servis gerektirmez.

---

## 🛠️ Kurulum ve Çalıştırma

### 1. Repository'yi Klonlayın:
```bash
git clone https://github.com/tarikmirzaomerli/PuffGuard.git
cd PuffGuard
```

### 2. Gerekli Kütüphaneleri Yükleyin:
```bash
pip install -r requirements.txt
```

### 3. Uygulamayı Başlatın:
```bash
python main.py
```
*(Windows kullanıcıları doğrudan `run_tracker.bat` dosyasına çift tıklayarak da başlatabilir).*

---

## 📁 Kayıt Konumu

Oluşturulan 10 saniyelik video kayıtları ve güvenlik fotoğrafları otomatik olarak proje dizinindeki **`cigarettes-foto`** klasörüne kaydedilir.
