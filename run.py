#!/usr/bin/env python3
"""
Sahne Değişikliği Tespiti - Ana Çalıştırma Scripti
"""

import argparse
import os
import sys
import subprocess
import json
from datetime import datetime


def run_training(args):
    """Eğitim çalıştır"""
    print(" Model eğitimi başlatılıyor...")
    
    cmd = [
        sys.executable, "train.py",
        "--model", args.model,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--dataset_path", args.dataset_path
    ]
    
    subprocess.run(cmd)


def run_testing(args):
    """Test çalıştır"""
    print(" Model testi başlatılıyor...")
    
    if not os.path.exists(args.model_path):
        print(f" Model dosyası bulunamadı: {args.model_path}")
        return
    
    cmd = [
        sys.executable, "test.py",
        "--model_path", args.model_path,
        "--dataset_path", args.dataset_path
    ]
    
    if args.config_path:
        cmd.extend(["--config_path", args.config_path])
    
    subprocess.run(cmd)


def run_inference(args):
    """Inference çalıştır"""
    print(" Inference başlatılıyor...")
    
    if not os.path.exists(args.model_path):
        print(f" Model dosyası bulunamadı: {args.model_path}")
        return
    
    cmd = [
        sys.executable, "inference.py",
        "--model_path", args.model_path,
        "--threshold", str(args.threshold)
    ]
    
    if args.single_image:
        cmd.extend(["--single_image", args.single_image])
    elif args.batch_dir:
        cmd.extend(["--batch_dir", args.batch_dir, "--output_dir", args.output_dir])
    elif args.live_camera:
        cmd.append("--live_camera")
    
    subprocess.run(cmd)


def setup_environment():
    """Çevre kurulumu"""
    print("Çevre kurulumu yapılıyor...")
    
    directories = [
        'checkpoints',
        'test_results',
        'inference_results',
        'runs',
        'logs',
        'configs',
        'visualizations'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"    {directory}/ oluşturuldu")
    
    print(" Çevre kurulumu tamamlandı!")


def show_system_info():
    """Sistem bilgilerini göster"""
    print(" Sistem Bilgileri:")
    print("-" * 40)
    
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("PyTorch yüklü değil!")
    
    try:
        import cv2
        print(f"OpenCV Version: {cv2.__version__}")
    except ImportError:
        print("OpenCV yüklü değil!")
    
    print("-" * 40)


def run_tensorboard():
    """TensorBoard başlat"""
    print(" TensorBoard başlatılıyor...")
    
    if not os.path.exists('runs'):
        os.makedirs('runs')
    
    cmd = ["tensorboard", "--logdir=runs", "--host=0.0.0.0", "--port=6006"]
    
    print(" TensorBoard URL: http://localhost:6006")
    print("   Durdurmak için Ctrl+C tuşlayın...")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n TensorBoard durduruldu.")


def run_dataset_check(dataset_path):
    """Dataset kontrolü"""
    print(f" Dataset kontrolü: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f" Dataset bulunamadı: {dataset_path}")
        return False
    
    input_dir = os.path.join(dataset_path, 'input')
    gt_dir = os.path.join(dataset_path, 'groundtruth')
    
    if not os.path.exists(input_dir):
        print(f" Input klasörü bulunamadı: {input_dir}")
        return False
    
    if not os.path.exists(gt_dir):
        print(f" Groundtruth klasörü bulunamadı: {gt_dir}")
        return False
    
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
    gt_files = [f for f in os.listdir(gt_dir) if f.endswith('.png')]
    
    print(f"    Input dosyaları: {len(input_files)}")
    print(f"    Groundtruth dosyaları: {len(gt_files)}")
    
    if len(input_files) != len(gt_files):
        print("  Input ve groundtruth dosya sayıları eşleşmiyor!")
        return False
    
    if len(input_files) == 0:
        print(" Hiç dosya bulunamadı!")
        return False
    
    print(" Dataset kontrolü başarılı!")
    return True


def create_sample_config():
    """Örnek konfigürasyon dosyası oluştur"""
    config = {
        "model_name": "unet",
        "dataset_path": "dataset",
        "num_epochs": 50,
        "batch_size": 8,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "optimizer": "adam",
        "bce_weight": 0.5,
        "dice_weight": 0.5,
        "train_split": 0.8,
        "num_workers": 2,
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 0.001
    }
    
    os.makedirs('configs', exist_ok=True)
    config_path = f"configs/sample_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f" Örnek konfigürasyon oluşturuldu: {config_path}")
    return config_path


def main():
    parser = argparse.ArgumentParser(
        description='Sahne Değişikliği Tespiti - Ana Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Kullanım Örnekleri:
  # Çevre kurulumu
  python run.py setup
  
  # Dataset kontrolü
  python run.py check-dataset --dataset_path dataset
  
  # Model eğitimi
  python run.py train --model unet --epochs 50
  
  # Model testi
  python run.py test --model_path checkpoints/best_model.pth
  
  # Tek görüntü inference
  python run.py inference --model_path checkpoints/best_model.pth --single_image image.jpg
  
  # TensorBoard
  python run.py tensorboard
  
  # Sistem bilgisi
  python run.py info
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Komutlar')
    
    subparsers.add_parser('setup', help='Çevre kurulumu')
    
    subparsers.add_parser('info', help='Sistem bilgilerini göster')
    
    dataset_parser = subparsers.add_parser('check-dataset', help='Dataset kontrolü')
    dataset_parser.add_argument('--dataset_path', type=str, default='dataset', help='Dataset yolu')
    
    train_parser = subparsers.add_parser('train', help='Model eğitimi')
    train_parser.add_argument('--model', type=str, default='unet', choices=['unet', 'resnet34', 'resnet50'])
    train_parser.add_argument('--epochs', type=int, default=50)
    train_parser.add_argument('--batch_size', type=int, default=8)
    train_parser.add_argument('--lr', type=float, default=0.001)
    train_parser.add_argument('--dataset_path', type=str, default='dataset')
    
    test_parser = subparsers.add_parser('test', help='Model testi')
    test_parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint yolu')
    test_parser.add_argument('--config_path', type=str, help='Konfigürasyon yolu')
    test_parser.add_argument('--dataset_path', type=str, default='dataset')
    
    inference_parser = subparsers.add_parser('inference', help='Model inference')
    inference_parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint yolu')
    inference_parser.add_argument('--threshold', type=float, default=0.5)
    inference_parser.add_argument('--single_image', type=str, help='Tek görüntü yolu')
    inference_parser.add_argument('--batch_dir', type=str, help='Toplu inference klasörü')
    inference_parser.add_argument('--output_dir', type=str, default='inference_results')
    inference_parser.add_argument('--live_camera', action='store_true', help='Canlı kamera')
    
    subparsers.add_parser('tensorboard', help='TensorBoard başlat')
    
    subparsers.add_parser('create-config', help='Örnek konfigürasyon oluştur')
    
    args = parser.parse_args()
    
    if args.command == 'setup':
        setup_environment()
    
    elif args.command == 'info':
        show_system_info()
    
    elif args.command == 'check-dataset':
        run_dataset_check(args.dataset_path)
    
    elif args.command == 'train':
        if run_dataset_check(args.dataset_path):
            run_training(args)
    
    elif args.command == 'test':
        run_testing(args)
    
    elif args.command == 'inference':
        run_inference(args)
    
    elif args.command == 'tensorboard':
        run_tensorboard()
    
    elif args.command == 'create-config':
        create_sample_config()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()