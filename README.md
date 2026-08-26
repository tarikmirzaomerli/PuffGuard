# PuffGuard - AI-Powered Smoking, Drowsiness & Obstruction Detection System 🚬😴🛡️🗣️

Laptop kamerası üzerinden gerçek zamanlı **sigara tüketimi**, **el-ağız hareketi**, **uyku/uyuşukluk (drowsiness)** ve **kamera güvenlik durumunu** takip eden; masaüstü bildirimi gönderip **Türkçe sesli ikaz** veren ve tespit anında otomatik video kaydı alan çok amaçlı hibrit yapay zeka sistemi.

---

## ✨ Özellikler

### 1. 🚬 Sigara Tespiti (İki Aşamalı Hibrit Model)
- **1. Aşama:** MediaPipe Face Mesh ile hafif ve CPU dostu dudak ve el takibi.
- **2. Aşama:** Ağız bölgesi etrafında çalışan yerel YOLOv8 ile yüksek doğruluklu sigara tespiti (`conf >= %65`).
- **Dinamik 2.2x ROI:** Yüzün kameraya olan mesafesine göre otomatik büyüyüp küçülen adaptif ağız kırpma alanı.
- **Gecikmeli Video Kaydı:** Sigara tespit edildiğinde olayın **5 saniye öncesini ve 5 saniye sonrasını** kapsayan toplam 10 saniyelik MP4 video kaydı (`cigarette_video_10s_YYYYMMDD_HHMMSS.mp4`).
- **15 Dakikalık (900 sn) Cooldown:** Üst üste bildirim yağmurunu önleyen bağımsız soğuma süresi.
- **🗣️ Sesli İkaz:** *"Sigara kullanımı tespit edildi. Kayıt alınıyor."*

### 2. 😴 Uyku & Uyuşukluk Tespiti (1 Dakika Kuralı & EAR Analizi)
- MediaPipe yüz nirengi noktaları üzerinden **EAR (Eye Aspect Ratio)** değerini gerçek zamanlı hesaplar (`EAR < 0.20` ise göz kapalı).
- **1 Dakikalık (60 Saniye) Kesintisiz Alarm:** Gözler aralıksız **1 dakika (60 saniye)** boyunca kapalı kaldığında **"UYARI: 1 Dakikadır Uyku Halindesiniz!"** masaüstü uyarısı gönderilir. Göz 1 dakika dolmadan açılırsa sayaç hemen sıfırlanır.
- **Kanıt Videosu:** Olay anının 10 saniyelik videosu `sleep_alert_YYYYMMDD_HHMMSS.mp4` adıyla kaydedilir.
- **2 Dakikalık (120 sn) Cooldown:** Uyku alarmları için bağımsız 2 dakikalık soğuma süresi.
- **🗣️ Sesli İkaz:** *"Lütfen uyanın! Bir dakikadır uyku halindesiniz."*

### 3. 🛡️ Kamera Güvenlik ve Engel Kontrolü (Canny Kenar Analizi)
- Kameranın kapatılması, kağıt/bezle engellenmesi veya karartılması durumunda anında alarm vermek yerine kesintisiz **1 dakika (60 saniye)** beklenir; engelleme sürerse **"UYARI: Kamera Önü Engellendi!"** bildirimi atılır ve son durum kaydı alınır.
- **Boş Oda / Masadan Kalkma:** Yüz görünmese bile odanın arka planı aydınlıksa sistem beklemede kalır, yanlış alarm vermez.
- **2 Dakikalık (120 sn) Cooldown:** Güvenlik alarmları için bağımsız 2 dakikalık soğuma süresi.
- **🗣️ Sesli İkaz:** *"Uyarı! Kamera görüşü engellendi."*

### 4. 🗣️ Yerel Türkçe Sesli İkaz Motoru (`pyttsx3`)
- İnternet gerektirmeyen, %100 yerel ve 150 WPM doğal konuşma hızında Türkçe sesli uyarılar.
- Kamera akışında sıfır gecikme (asenkron threading).

### 5. ⚡ %100 Ücretsiz & Yerel
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
