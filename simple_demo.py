"""
Basit Demo - Manuel Komutlarla Sahne Değişikliği Tespiti
"""

import os
import subprocess
import sys

def run_simple_demo():
    """
    Basit demo - adım adım manuel komutlar
    """
    print(" BASIT SAHNE DEĞİŞİKLİĞİ TESPİTİ DEMO")
    print("=" * 45)
    
    
    model_path = find_best_model()
    if not model_path:
        print(" Model dosyası bulunamadı!")
        print("Önce eğitim yapın: python train.py --model unet --epochs 10")
        return False
    
    print(f" Model bulundu: {model_path}")
    
    
    input_dir = "dataset/input"
    if not os.path.exists(input_dir):
        print(f" Input klasörü bulunamadı: {input_dir}")
        return False
    
    
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.jpg')])[:3]
    
    if not image_files:
        print(" Test edilecek görüntü bulunamadı!")
        return False
    
    print(f"📸 {len(image_files)} görüntü test edilecek:")
    for i, img in enumerate(image_files, 1):
        print(f"   {i}. {img}")
    
    print("\n" + "─" * 45)
    input("Devam etmek için Enter'a basın...")
    
    
    for i, image_file in enumerate(image_files, 1):
        print(f"\n  TEST {i}/{len(image_files)}: {image_file}")
        print("─" * 30)
        
        image_path = os.path.join(input_dir, image_file).replace('\\', '/')
        
        
        cmd = f'python inference.py --model_path "{model_path}" --single_image "{image_path}" --threshold 0.5'
        
        print(f"Çalıştırılan komut:")
        print(f"   {cmd}")
        print()
        
        try:
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                output = result.stdout
                print(" Başarılı!")
                
                
                for line in output.split('\n'):
                    if "Değişiklik Yüzdesi:" in line:
                        percentage = line.split(':')[1].strip()
                        print(f"    Değişiklik: {percentage}")
                    elif "Güven Skoru:" in line:
                        confidence = line.split(':')[1].strip()
                        print(f"    Güven: {confidence}")
                
                
                if "inference_results" in output:
                    print("    Görsel sonuçlar oluşturuldu!")
                
            else:
                print(" Hata oluştu:")
                print(f"   Return code: {result.returncode}")
                print(f"   Stderr: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(" Zaman aşımı - 60 saniye geçti")
        except Exception as e:
            print(f" Beklenmeyen hata: {e}")
    
    print("\n" + "=" * 45)
    print(" DEMO TAMAMLANDI!")
    
    
    show_results_summary()
    
    return True

def find_best_model():
    """En iyi model dosyasını bul"""
    model_candidates = [
        "checkpoints/best_model.pth",
        "checkpoints/last_checkpoint.pth"
    ]
    
    for model_path in model_candidates:
        if os.path.exists(model_path):
            return model_path
    
    
    if os.path.exists("checkpoints"):
        pth_files = [f for f in os.listdir("checkpoints") if f.endswith('.pth')]
        if pth_files:
            latest_file = max(pth_files, key=lambda x: os.path.getctime(os.path.join("checkpoints", x)))
            return os.path.join("checkpoints", latest_file)
    
    return None

def show_results_summary():
    """Sonuç özeti göster"""
    print("\n SONUÇ ÖZETİ:")
    print("─" * 25)
    
    
    model_path = find_best_model()
    if model_path:
        model_size = os.path.getsize(model_path) / (1024*1024)
        print(f" Model: {os.path.basename(model_path)} ({model_size:.1f} MB)")
    
    
    if os.path.exists("inference_results"):
        files = os.listdir("inference_results")
        prediction_files = [f for f in files if "_prediction.png" in f]
        overlay_files = [f for f in files if "_overlay.png" in f]
        report_files = [f for f in files if "_report.txt" in f]
        
        print(f" Görsel sonuçlar: {len(prediction_files)} adet")
        print(f" Overlay'ler: {len(overlay_files)} adet")  
        print(f" Raporlar: {len(report_files)} adet")
        
        if prediction_files:
            print(f"\n Sonuçları görmek için: inference_results/ klasörüne bakın")
            print("   Dosya türleri:")
            print("   - *_prediction.png : Değişiklik heatmap'i")
            print("   - *_overlay.png    : Kırmızı değişiklik overlay'i")  
            print("   - *_mask.png       : Binary maske")
            print("   - *_report.txt     : Detaylı rapor")
    
   
    if os.path.exists("runs"):
        run_dirs = [d for d in os.listdir("runs") if os.path.isdir(os.path.join("runs", d))]
        if run_dirs:
            print(f"TensorBoard logları: {len(run_dirs)} eğitim")
            print("   Görüntülemek için: tensorboard --logdir=runs")
    
    print("\n Sahne değişikliği tespiti sistemi çalışıyor!")

def quick_inference_test():
    """Hızlı tek görüntü testi"""
    print(" HIZLI TEK GÖRÜNTÜ TESTİ")
    print("=" * 30)
    
    model_path = find_best_model()
    if not model_path:
        print(" Model bulunamadı!")
        return
    
    
    test_image = "dataset/input/in000001.jpg"
    if not os.path.exists(test_image):
        print(f" Test görüntüsü bulunamadı: {test_image}")
        return
    
    print(f"  Test görüntüsü: {test_image}")
    print(f" Model: {model_path}")
    
    cmd = f'python inference.py --model_path "{model_path}" --single_image "{test_image}"'
    print(f"\n Komut: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, timeout=30)
        if result.returncode == 0:
            print(" Test başarılı!")
        else:
            print(f" Test başarısız (code: {result.returncode})")
    except subprocess.TimeoutExpired:
        print(" Zaman aşımı")
    except Exception as e:
        print(f" Hata: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_inference_test()
    else:
        run_simple_demo()