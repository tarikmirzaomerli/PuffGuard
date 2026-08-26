# PuffGuard - AI-Powered Smoking Detection & Notification System 🚬🛡️

Laptop kamerası üzerinden gerçek zamanlı el-ağız hareketi ve sigara tespiti yapan, masaüstü bildirimi gönderen ve tespit anında otomatik video kaydı alan iki aşamalı hibrit yapay zeka uygulaması.

---

## ✨ Özellikler

- **İki Aşamalı Hibrit Tespit:**
  - **1. Aşama:** MediaPipe ile hafif ve CPU dostu el-ağız mesafesi ve yüz takibi.
  - **2. Aşama:** Dudak bölgesi etrafında çalışan yerel YOLOv8 ile yüksek doğruluklu sigara tespiti.
- **Akıllı Zaman Doğrulaması:** Yanlış alarmları önlemek için 12 karelik kesintisiz doğrulama mantığı.
- **Gecikmeli Video Kaydı:** Sigara tespit edildiğinde olayın **5 saniye öncesini ve 5 saniye sonrasını** kapsayan toplam 10 saniyelik MP4 video kaydı (`collections.deque` tamponu).
- **Masaüstü Bildirimi & Soğuma Süresi:** Windows Bildirim Merkezi uyarısı ve üst üste bildirim yağmurunu önleyen **15 dakikalık (900 sn)** soğuma süresi.
- **Kamera Güvenlik Kontrolü:** Kameranın kapatılması, kağıt/bezle engellenmesi veya karartılması durumunda otomatik uyarı ve son durum kaydı.
- **%100 Ücretsiz & Yerel:** Tüm işlemler yerel cihaz üzerinde çalışır, harici bulut API'si veya ücretli servis gerektirmez.

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

Oluşturulan 10 saniyelik video kayıtları (`cigarette_video_10s_YYYYMMDD_HHMMSS.mp4`) ve güvenlik fotoğrafları otomatik olarak proje dizinindeki **`cigarettes-foto`** klasörüne kaydedilir.
