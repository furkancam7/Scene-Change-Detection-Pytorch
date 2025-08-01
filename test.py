"""
Eğitilmiş modelin test edilmesi ve görselleştirme
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import json
from tqdm import tqdm

from model import get_model
from dataset import SceneChangeDataset, get_data_transforms
from utils import calculate_metrics, visualize_predictions


class SceneChangeTester:
    """
    Eğitilmiş model test sınıfı
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
            
            self.config = {
                'model_name': 'unet',
                'dataset_path': 'dataset'
            }
        
        
        self.model = get_model(
            model_name=self.config['model_name'],
            n_channels=3,
            n_classes=1
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model yüklendi: {self.config['model_name']}")
        print(f"Epoch: {checkpoint.get('epoch', 'Bilinmiyor')}")
        print(f"En iyi validation loss: {checkpoint.get('best_val_loss', 'Bilinmiyor')}")
        
        
        _, self.transform, self.target_transform = get_data_transforms()
    
    def test_on_dataset(self, dataset_path, save_results=True):
        """
        Tüm dataset üzerinde test
        """
        
        dataset = SceneChangeDataset(
            dataset_path=dataset_path,
            transform=self.transform,
            target_transform=self.target_transform
        )
        
        
        test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=2
        )
        
        all_predictions = []
        all_targets = []
        all_indices = []
        
        print("Dataset üzerinde test yapılıyor...")
        
        with torch.no_grad():
            for batch_idx, (images, targets, indices) in enumerate(tqdm(test_loader)):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                
                predictions = self.model(images)
                
                
                predictions = predictions.cpu().numpy()
                targets = targets.cpu().numpy()
                images = images.cpu().numpy()
                
                all_predictions.append(predictions)
                all_targets.append(targets)
                all_indices.extend(indices.tolist())
                
                
                if save_results and batch_idx < 5:
                    visualize_predictions(
                        images, predictions, targets, indices,
                        save_dir=f'test_results/visualizations'
                    )
        
        
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        
        metrics = calculate_metrics(all_predictions, all_targets)
        
        print("\nTest Sonuçları:")
        print(f"IoU: {metrics['iou']:.4f}")
        print(f"Dice: {metrics['dice']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        
        if save_results:
            os.makedirs('test_results', exist_ok=True)
            
            with open('test_results/test_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            
            self.plot_test_results(all_predictions, all_targets)
        
        return metrics, all_predictions, all_targets
    
    def test_single_image(self, image_path, save_result=True):
        """
        Tek görüntü üzerinde test
        """
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(input_tensor)
            prediction = prediction.cpu().numpy().squeeze()
        
        if save_result:
            self.visualize_single_result(image, prediction, image_path, original_size)
        
        return prediction
    
    def visualize_single_result(self, original_image, prediction, image_path, original_size):
        """
        Tek görüntü sonucunu görselleştir
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(prediction, cmap='hot')
        axes[1].set_title('Change Detection Result')
        axes[1].axis('off')
        
        prediction_resized = np.array(Image.fromarray((prediction * 255).astype(np.uint8)).resize(original_size))
        prediction_resized = prediction_resized / 255.0
        
        img_array = np.array(original_image) / 255.0
        overlay = img_array.copy()
        
        change_mask = prediction_resized > 0.5
        overlay[change_mask] = [1, 0, 0]
        
        result = 0.7 * img_array + 0.3 * overlay
        
        axes[2].imshow(result)
        axes[2].set_title('Change Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        os.makedirs('test_results/single_tests', exist_ok=True)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = f'test_results/single_tests/{image_name}_result.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Sonuç kaydedildi: {save_path}")
    
    def plot_test_results(self, predictions, targets):
        """
        Test sonuçlarının grafiklerini çiz
        """
        thresholds = np.arange(0.1, 1.0, 0.1)
        ious = []
        f1s = []
        precisions = []
        recalls = []
        
        for thresh in thresholds:
            metrics = calculate_metrics(predictions, targets, threshold=thresh)
            ious.append(metrics['iou'])
            f1s.append(metrics['f1'])
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].plot(thresholds, ious, 'b-o')
        axes[0, 0].set_title('IoU vs Threshold')
        axes[0, 0].set_xlabel('Threshold')
        axes[0, 0].set_ylabel('IoU')
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(thresholds, f1s, 'g-o')
        axes[0, 1].set_title('F1 Score vs Threshold')
        axes[0, 1].set_xlabel('Threshold')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(thresholds, precisions, 'r-o', label='Precision')
        axes[1, 0].plot(thresholds, recalls, 'orange', marker='o', label='Recall')
        axes[1, 0].set_title('Precision & Recall vs Threshold')
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(recalls, precisions, 'purple', marker='o')
        axes[1, 1].set_title('Precision-Recall Curve')
        axes[1, 1].set_xlabel('Recall')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('test_results/test_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Test analiz grafikleri kaydedildi: test_results/test_analysis.png")
    
    def compare_models(self, model_paths, dataset_path):
        """
        Birden fazla modeli karşılaştır
        """
        results = {}
        
        for model_path in model_paths:
            model_name = os.path.basename(model_path).replace('.pth', '')
            print(f"\n{model_name} test ediliyor...")
            
            tester = SceneChangeTester(model_path)
            
            metrics, _, _ = tester.test_on_dataset(dataset_path, save_results=False)
            results[model_name] = metrics
        
        print("\nModel Karşılaştırması:")
        print("-" * 80)
        print(f"{'Model':<20} {'IoU':<8} {'Dice':<8} {'F1':<8} {'Precision':<12} {'Recall':<8}")
        print("-" * 80)
        
        for model_name, metrics in results.items():
            print(f"{model_name:<20} {metrics['iou']:<8.4f} {metrics['dice']:<8.4f} "
                  f"{metrics['f1']:<8.4f} {metrics['precision']:<12.4f} {metrics['recall']:<8.4f}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Sahne Değişikliği Tespiti Model Test')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint yolu')
    parser.add_argument('--config_path', type=str, help='Konfigürasyon dosyası yolu')
    parser.add_argument('--dataset_path', type=str, default='dataset', help='Test dataset yolu')
    parser.add_argument('--single_image', type=str, help='Tek görüntü test için dosya yolu')
    parser.add_argument('--compare_models', nargs='+', help='Karşılaştırılacak model yolları')
    
    args = parser.parse_args()
    
    if args.compare_models:
        tester = SceneChangeTester(args.compare_models[0])
        tester.compare_models(args.compare_models, args.dataset_path)
    
    elif args.single_image:
        print(f"Tek görüntü testi: {args.single_image}")
        tester = SceneChangeTester(args.model_path, args.config_path)
        prediction = tester.test_single_image(args.single_image)
        
        change_percentage = (prediction > 0.5).mean() * 100
        print(f"Değişiklik yüzdesi: {change_percentage:.2f}%")
    
    else:
        print(f"Dataset testi: {args.dataset_path}")
        tester = SceneChangeTester(args.model_path, args.config_path)
        metrics, predictions, targets = tester.test_on_dataset(args.dataset_path)
        
        change_pixels_gt = (targets > 0.5).sum()
        change_pixels_pred = (predictions > 0.5).sum()
        total_pixels = targets.size
        
        print(f"\nÖzet İstatistikler:")
        print(f"Toplam piksel: {total_pixels:,}")
        print(f"Ground Truth'ta değişiklik pikseli: {change_pixels_gt:,} ({change_pixels_gt/total_pixels*100:.2f}%)")
        print(f"Tahmin edilen değişiklik pikseli: {change_pixels_pred:,} ({change_pixels_pred/total_pixels*100:.2f}%)")


if __name__ == "__main__":
    main()