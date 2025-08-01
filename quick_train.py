"""
Hızlı test eğitimi - Parametreler CPU için optimize edilmiş
"""

import subprocess
import sys

def run_quick_training():
    """
    Hızlı test eğitimi çalıştır
    """
    print(" Hızlı test eğitimi başlatılıyor...")
    print(" Optimize edilmiş parametreler:")
    print("   - Epochs: 5 (hızlı test için)")
    print("   - Batch Size: 4 (CPU için optimal)")
    print("   - Learning Rate: 0.01 (hızlı öğrenme)")
    print("   - Model: unet (en hafif)")
    print("")
    
    cmd = [
        sys.executable, "train.py",
        "--model", "unet",
        "--epochs", "2",
        "--batch_size", "4",
        "--lr", "0.01"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("\n Hızlı test tamamlandı!")
        print(" Sonuçlar:")
        print("   - Model: checkpoints/best_model.pth")
        print("   - Loglar: runs/ klasöründe")
        print("   - Grafikleri görmek için: tensorboard --logdir=runs")
        
    except subprocess.CalledProcessError as e:
        print(f"\n Eğitim hatası: {e}")
        return False
    except KeyboardInterrupt:
        print("\n  Eğitim kullanıcı tarafından durduruldu")
        return False
    
    return True

if __name__ == "__main__":
    success = run_quick_training()
    
    if success:
        print("\n Test önerisi:")
        print("python test.py --model_path checkpoints/best_model.pth")
        
        print("\n Inference önerisi:")
        print("python inference.py --model_path checkpoints/best_model.pth --single_image dataset/input/in000001.jpg")