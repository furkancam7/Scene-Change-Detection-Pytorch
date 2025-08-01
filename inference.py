"""
Eğitilmiş model ile inference (tahmin) yapma
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import json
import cv2
from pathlib import Path

from model import get_model
from dataset import get_data_transforms


class SceneChangeInference:
    """
    Eğitilmiş model ile inference sınıfı
    """
    
    def __init__(self, model_path, config_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Kullanılan cihaz: {self.device}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        elif 'config' in checkpoint:
            self.config = checkpoint['config']
        else:
            self.config = {'model_name': 'unet'}
        

        self.model = get_model(
            model_name=self.config['model_name'],
            n_channels=3,
            n_classes=1
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model yüklendi: {self.config['model_name']}")
        
        _, self.transform, _ = get_data_transforms()
    
    def predict_single_image(self, image_path, threshold=0.5):
        """
        Tek görüntü üzerinde tahmin yap
        
        Args:
            image_path: Görüntü dosya yolu
            threshold: Binary threshold değeri
            
        Returns:
            prediction: Tahmin maskesi
            confidence: Güven skoru
        """
        
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
       
        input_tensor = self.transform(image)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        
        
        with torch.no_grad():
            prediction = self.model(input_tensor)
            prediction = torch.sigmoid(prediction)  # Eğer sigmoid yok ise
            prediction = prediction.cpu().numpy().squeeze()
        
        
        binary_mask = (prediction > threshold).astype(np.uint8)
        
       
        confidence = np.mean(np.abs(prediction - 0.5)) * 2  # 0-1 arası
        
        prediction_resized = cv2.resize(prediction, original_size, interpolation=cv2.INTER_LINEAR)
        binary_mask_resized = cv2.resize(binary_mask, original_size, interpolation=cv2.INTER_NEAREST)
        
        return prediction_resized, binary_mask_resized, confidence
    
    def predict_image_pair(self, image1_path, image2_path, threshold=0.5):
        """
        İki görüntü arasındaki farkı tespit et
        
        Args:
            image1_path: İlk görüntü yolu
            image2_path: İkinci görüntü yolu
            threshold: Binary threshold
            
        Returns:
            change_map: Değişiklik haritası
            overlay: Görsel overlay
        """
        img1 = Image.open(image1_path).convert('RGB')
        img2 = Image.open(image2_path).convert('RGB')
        
        size = img1.size
        img2 = img2.resize(size)
        
        diff_img = Image.fromarray(np.abs(np.array(img1) - np.array(img2)))
        
        prediction, binary_mask, confidence = self.predict_single_image(
            image1_path,  
            threshold
        )
        
        
        overlay = self.create_change_overlay(img2, binary_mask)
        
        return prediction, binary_mask, overlay, confidence
    
    def create_change_overlay(self, original_image, change_mask, alpha=0.3):
        """
        Değişiklik overlay'i oluştur
        
        Args:
            original_image: Orijinal görüntü (PIL Image)
            change_mask: Binary değişiklik maskesi
            alpha: Overlay şeffaflığı
            
        Returns:
            overlay_image: Overlay'li görüntü
        """
        img_array = np.array(original_image) / 255.0
        
        overlay = np.zeros_like(img_array)
        overlay[change_mask == 1] = [1, 0, 0]  
        
        result = (1 - alpha) * img_array + alpha * overlay
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        
        return Image.fromarray(result)
    
    def batch_inference(self, input_dir, output_dir, threshold=0.5):
        """
        Klasördeki tüm görüntüler için toplu tahmin
        
        Args:
            input_dir: Giriş klasörü
            output_dir: Çıkış klasörü
            threshold: Binary threshold
        """
        os.makedirs(output_dir, exist_ok=True)
        
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        image_files = []
        for ext in supported_formats:
            image_files.extend(Path(input_dir).glob(f'*{ext}'))
            image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
        
        print(f"{len(image_files)} görüntü bulundu")
        
        for i, image_path in enumerate(image_files):
            print(f"İşleniyor: {image_path.name} ({i+1}/{len(image_files)})")
            
            try:
                
                prediction, binary_mask, confidence = self.predict_single_image(
                    str(image_path), threshold
                )
                
                
                base_name = image_path.stem
                
                
                plt.figure(figsize=(10, 5))
                
                plt.subplot(1, 2, 1)
                original_img = Image.open(image_path)
                plt.imshow(original_img)
                plt.title('Original Image')
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.imshow(prediction, cmap='hot')
                plt.title(f'Change Detection (Confidence: {confidence:.3f})')
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{base_name}_prediction.png'), 
                           dpi=150, bbox_inches='tight')
                plt.close()
                
                
                cv2.imwrite(os.path.join(output_dir, f'{base_name}_mask.png'), 
                           binary_mask * 255)
                
                
                overlay = self.create_change_overlay(original_img, binary_mask)
                overlay.save(os.path.join(output_dir, f'{base_name}_overlay.png'))
                
                
                change_percentage = (binary_mask.sum() / binary_mask.size) * 100
                
                
                with open(os.path.join(output_dir, f'{base_name}_report.txt'), 'w') as f:
                    f.write(f"Görüntü: {image_path.name}\n")
                    f.write(f"Değişiklik Yüzdesi: {change_percentage:.2f}%\n")
                    f.write(f"Güven Skoru: {confidence:.3f}\n")
                    f.write(f"Threshold: {threshold}\n")
                    f.write(f"Toplam Piksel: {binary_mask.size:,}\n")
                    f.write(f"Değişiklik Pikseli: {binary_mask.sum():,}\n")
                
            except Exception as e:
                print(f"Hata: {image_path.name} - {str(e)}")
                continue
        
        print(f"Toplu inference tamamlandı. Sonuçlar: {output_dir}")
    
    def live_inference(self, camera_index=0):
        """
        Kameradan canlı inference
        
        Args:
            camera_index: Kamera indeksi
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("Kamera açılamadı")
            return
        
        print("Canlı inference başlatıldı. 'q' tuşuna basarak çıkabilirsiniz.")
        
        reference_frame = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if reference_frame is None:
                reference_frame = frame.copy()
                continue
            
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            temp_path = 'temp_frame.jpg'
            frame_pil.save(temp_path)
            
            try:
                prediction, binary_mask, confidence = self.predict_single_image(temp_path)
                
                overlay = self.create_change_overlay(frame_pil, binary_mask)
                overlay_cv = cv2.cvtColor(np.array(overlay), cv2.COLOR_RGB2BGR)
                
                cv2.putText(overlay_cv, f'Confidence: {confidence:.3f}', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                change_percentage = (binary_mask.sum() / binary_mask.size) * 100
                cv2.putText(overlay_cv, f'Change: {change_percentage:.1f}%', 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.imshow('Scene Change Detection', overlay_cv)
                
            except Exception as e:
                print(f"Frame işleme hatası: {e}")
                cv2.imshow('Scene Change Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        if os.path.exists('temp_frame.jpg'):
            os.remove('temp_frame.jpg')


def main():
    parser = argparse.ArgumentParser(description='Sahne Değişikliği Tespiti - Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint yolu')
    parser.add_argument('--config_path', type=str, help='Konfigürasyon dosyası yolu')
    parser.add_argument('--threshold', type=float, default=0.5, help='Binary threshold (0-1)')
    
    parser.add_argument('--single_image', type=str, help='Tek görüntü inference')
    parser.add_argument('--image_pair', nargs=2, help='İki görüntü karşılaştırması')
    parser.add_argument('--batch_dir', type=str, help='Toplu inference için klasör')
    parser.add_argument('--output_dir', type=str, default='inference_results', help='Çıkış klasörü')
    parser.add_argument('--live_camera', action='store_true', help='Canlı kamera inference')
    parser.add_argument('--camera_index', type=int, default=0, help='Kamera indeksi')
    
    args = parser.parse_args()
    
    inferencer = SceneChangeInference(args.model_path, args.config_path)
    
    if args.single_image:
        print(f"Tek görüntü inference: {args.single_image}")
        prediction, binary_mask, confidence = inferencer.predict_single_image(
            args.single_image, args.threshold
        )
        
        change_percentage = (binary_mask.sum() / binary_mask.size) * 100
        print(f"Sonuçlar:")
        print(f"   Değişiklik Yüzdesi: {change_percentage:.2f}%")
        print(f"   Güven Skoru: {confidence:.3f}")
        
        original_img = Image.open(args.single_image)
        overlay = inferencer.create_change_overlay(original_img, binary_mask)
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(original_img)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(prediction, cmap='hot')
        plt.title(f'Change Heatmap (Confidence: {confidence:.3f})')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(overlay)
        plt.title(f'Change Overlay ({change_percentage:.1f}% change)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    elif args.image_pair:
        img1_path, img2_path = args.image_pair
        print(f"Görüntü karşılaştırması: {img1_path} vs {img2_path}")
        
        prediction, binary_mask, overlay, confidence = inferencer.predict_image_pair(
            img1_path, img2_path, args.threshold
        )
        
        change_percentage = (binary_mask.sum() / binary_mask.size) * 100
        print(f"Karşılaştırma Sonuçları:")
        print(f"   Değişiklik Yüzdesi: {change_percentage:.2f}%")
        print(f"   Güven Skoru: {confidence:.3f}")
        
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        
        plt.figure(figsize=(20, 5))
        
        plt.subplot(1, 4, 1)
        plt.imshow(img1)
        plt.title('Image 1')
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.imshow(img2)
        plt.title('Image 2')
        plt.axis('off')
        
        plt.subplot(1, 4, 3)
        plt.imshow(prediction, cmap='hot')
        plt.title('Change Heatmap')
        plt.axis('off')
        
        plt.subplot(1, 4, 4)
        plt.imshow(overlay)
        plt.title(f'Change Overlay ({change_percentage:.1f}%)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    elif args.batch_dir:
        print(f"Toplu inference: {args.batch_dir}")
        inferencer.batch_inference(args.batch_dir, args.output_dir, args.threshold)
    
    elif args.live_camera:
        print(f"Canlı kamera inference başlatılıyor...")
        inferencer.live_inference(args.camera_index)
    
    else:
        print("Bir inference modu seçmelisiniz!")
        parser.print_help()


if __name__ == "__main__":
    main()