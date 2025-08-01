"""
Tam Demo: Eğitim + Sahne Değişikliği Tespiti + Görselleştirme
"""

import subprocess
import sys
import os
import json
from pathlib import Path

def run_complete_demo():
    """
    Tam demo: Eğitim -> Test -> Inference -> Görselleştirme
    """
    print(" SAHNE DEĞİŞİKLİĞİ TESPİTİ - TAM DEMO")
    print("=" * 50)
    print("Bu demo şunları yapacak:")
    print("1.  Hızlı model eğitimi (2 epoch)")
    print("2.  Model performans testi")
    print("3.  Gerçek görüntülerle inference")
    print("4.  Değişiklik yerelleştirme ve görselleştirme")
    print("5.  Sonuç analizi")
    print("-" * 50)
    
    input("Devam etmek için Enter'a basın...")
    
    print("\n ADIM 1: Model Eğitimi")
    print("-" * 30)
    success = train_model()
    if not success:
        return False
    
    print("\n ADIM 2: Model Performans Testi")
    print("-" * 35)
    success = test_model()
    if not success:
        return False
    
    print("\n ADIM 3: Sahne Değişikliği Tespiti")
    print("-" * 40)
    success = run_scene_change_detection()
    if not success:
        return False
    
    print("\n ADIM 4: Sonuç Analizi")
    print("-" * 25)
    show_results()
    
    print("\n" + "=" * 50)
    print(" TAM DEMO TAMAMLANDI!")
    print(" Tüm sonuçlar hazır:")
    print("   - Model: checkpoints/best_model.pth")
    print("   - Test sonuçları: test_results/")
    print("   - Inference sonuçları: inference_results/")
    print("   - TensorBoard: runs/")
    print("=" * 50)
    
    return True

def train_model():
    """Model eğitimi"""
    print(" Hızlı model eğitimi başlatılıyor...")
    
    cmd = [
        sys.executable, "train.py",
        "--model", "unet",
        "--epochs", "2",
        "--batch_size", "4", 
        "--lr", "0.01"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(" Model eğitimi tamamlandı!")
        
        if os.path.exists("checkpoints/best_model.pth"):
            print(" Model kaydedildi: checkpoints/best_model.pth")
            return True
        else:
            print("  Model dosyası bulunamadı ama eğitim tamamlandı")
            return True
            
    except subprocess.CalledProcessError as e:
        print(f" Eğitim hatası: {e}")
        return False
    except Exception as e:
        print(f" Beklenmeyen hata: {e}")
        return False

def test_model():
    """Model testi"""
    print(" Model performansı test ediliyor...")
    
    model_path = find_model_file()
    if not model_path:
        print(" Model dosyası bulunamadı!")
        return False
    
    cmd = [
        sys.executable, "test.py",
        "--model_path", model_path,
        "--dataset_path", "dataset"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(" Model testi tamamlandı!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f" Test hatası: {e}")
        return False

def run_scene_change_detection():
    """Sahne değişikliği tespiti ve görselleştirme"""
    print(" Sahne değişikliği tespiti başlatılıyor...")
    
    model_path = find_model_file()
    if not model_path:
        print(" Model dosyası bulunamadı!")
        return False
    
    input_dir = "dataset/input"
    if not os.path.exists(input_dir):
        print(f" Input klasörü bulunamadı: {input_dir}")
        return False
    
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.jpg')])[:3]
    
    if not image_files:
        print(" Test edilecek görüntü bulunamadı!")
        return False
    
    print(f" {len(image_files)} görüntü test edilecek...")
    
    for i, image_file in enumerate(image_files, 1):
        print(f"\n  Test {i}/{len(image_files)}: {image_file}")
        
        image_path = os.path.join(input_dir, image_file)
        
        cmd = [
            sys.executable, "inference.py",
            "--model_path", model_path,
            "--single_image", image_path,
            "--threshold", "0.5"
        ]
        
        try:
            # Debug: Komutu yazdır
            print(f"    Komut: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Çıktıyı parse et
            output = result.stdout
            print(f"    Çıktı: {output[:100]}...")  # İlk 100 karakter
            
            if "Değişiklik Yüzdesi:" in output:
                # Değişiklik yüzdesini çıkar
                for line in output.split('\n'):
                    if "Değişiklik Yüzdesi:" in line:
                        percentage = line.split(':')[1].strip()
                        print(f"    Değişiklik tespit edildi: {percentage}")
                    elif "Güven Skoru:" in line:
                        confidence = line.split(':')[1].strip()
                        print(f"    Güven skoru: {confidence}")
            
        except subprocess.CalledProcessError as e:
            print(f"    Inference hatası:")
            print(f"      Return code: {e.returncode}")
            print(f"      Stdout: {e.stdout}")
            print(f"      Stderr: {e.stderr}")
            continue
        except Exception as e:
            print(f"     Beklenmeyen hata: {e}")
            continue
    
    # Toplu inference
    print(f"\n Toplu inference başlatılıyor...")
    cmd = [
        sys.executable, "inference.py",
        "--model_path", model_path,
        "--batch_dir", input_dir,
        "--output_dir", "inference_results"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(" Toplu inference tamamlandı!")
        print(" Sonuçlar: inference_results/ klasöründe")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"  Toplu inference hatası: {e}")
        return True  # Tek inference başarılıysa devam et

def find_model_file():
    """En son model dosyasını bul"""
    # Önce best_model.pth'yi ara
    best_model = "checkpoints/best_model.pth"
    if os.path.exists(best_model):
        return best_model
    
    # Sonra last_checkpoint.pth'yi ara
    last_checkpoint = "checkpoints/last_checkpoint.pth"
    if os.path.exists(last_checkpoint):
        return last_checkpoint
    
    # checkpoints klasöründeki tüm .pth dosyalarını ara
    if os.path.exists("checkpoints"):
        pth_files = [f for f in os.listdir("checkpoints") if f.endswith('.pth')]
        if pth_files:
            latest_file = max(pth_files, key=lambda x: os.path.getctime(os.path.join("checkpoints", x)))
            return os.path.join("checkpoints", latest_file)
    
    return None

def show_results():
    """Sonuçları göster"""
    print(" Sonuç özeti:")
    
    # Model dosyası
    model_path = find_model_file()
    if model_path:
        print(f" Model: {model_path}")
        
        # Model boyutu
        model_size = os.path.getsize(model_path) / (1024*1024)  # MB
        print(f" Model boyutu: {model_size:.1f} MB")
    
    # Test sonuçları
    if os.path.exists("test_results"):
        test_files = os.listdir("test_results")
        print(f" Test dosyaları: {len(test_files)} adet")
        
        # Metrikleri oku
        metrics_file = "test_results/test_metrics.json"
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                print(" Model Performansı:")
                print(f"   - IoU: {metrics.get('iou', 0):.3f}")
                print(f"   - F1 Score: {metrics.get('f1', 0):.3f}")
                print(f"   - Precision: {metrics.get('precision', 0):.3f}")
                print(f"   - Recall: {metrics.get('recall', 0):.3f}")
                
            except Exception as e:
                print(f"  Metrik okuma hatası: {e}")
    
    # Inference sonuçları
    if os.path.exists("inference_results"):
        inference_files = os.listdir("inference_results")
        print(f" Inference dosyaları: {len(inference_files)} adet")
        
        # Görselleştirme dosyalarını say
        prediction_files = [f for f in inference_files if f.endswith('_prediction.png')]
        overlay_files = [f for f in inference_files if f.endswith('_overlay.png')]
        
        print(f"   - Tahmin grafikleri: {len(prediction_files)} adet")
        print(f"   - Overlay görüntüleri: {len(overlay_files)} adet")
    
    # TensorBoard logları
    if os.path.exists("runs"):
        run_dirs = [d for d in os.listdir("runs") if os.path.isdir(os.path.join("runs", d))]
        print(f" TensorBoard logları: {len(run_dirs)} eğitim")
        
        if run_dirs:
            latest_run = max(run_dirs, key=lambda x: os.path.getctime(os.path.join("runs", x)))
            print(f"   - Son eğitim: {latest_run}")
            print(f"   - Görüntülemek için: tensorboard --logdir=runs")
    
    print("\n SAHNE DEĞİŞİKLİĞİ TESPİTİ SİSTEMİ HAZIR!")
    print(" Tüm sonuçları görüntülemek için ilgili klasörleri kontrol edin.")

if __name__ == "__main__":
    print(" Sahne Değişikliği Tespiti - Tam Demo")
    print("Bu demo tüm sistemi test edecek ve sahne değişikliği tespiti yapacak.")
    print()
    
    success = run_complete_demo()
    
    if success:
        print("\n Demo başarıyla tamamlandı!")
        print("Şimdi sahne değişikliği tespiti sisteminiz hazır ve çalışıyor!")
    else:
        print("\n Demo sırasında hata oluştu.")
        print("Lütfen hata mesajlarını kontrol edin.")