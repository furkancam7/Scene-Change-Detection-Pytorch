"""
Sahne Değişikliği Tespiti için Model Mimarisi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConvBlock(nn.Module):
    """
    @param in_channels: Giriş kanal sayısı
    @param out_channels: Çıkış kanal sayısı
    @param kernel_size: Kernel boyutu
    @param padding: Padding değeri
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DoubleConvBlock(nn.Module):
    """
    @param in_channels: Giriş kanal sayısı
    @param out_channels: Çıkış kanal sayısı
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConvBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SceneChangeUNet(nn.Module):
    """
    @param n_channels: Giriş kanal sayısı (varsayılan: 3 RGB)
    @param n_classes: Çıkış sınıf sayısı (varsayılan: 1 binary)
    """
    def __init__(self, n_channels=3, n_classes=1):
        super(SceneChangeUNet, self).__init__()
        
        self.encoder1 = DoubleConvBlock(n_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.encoder2 = DoubleConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.encoder3 = DoubleConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.encoder4 = DoubleConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        self.bottleneck = DoubleConvBlock(512, 1024)
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.decoder4 = DoubleConvBlock(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.decoder3 = DoubleConvBlock(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder2 = DoubleConvBlock(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder1 = DoubleConvBlock(128, 64)
        
        self.final_conv = nn.Conv2d(64, n_classes, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        @param x: Input tensor (N, C, H, W)
        @return: Output segmentation mask (N, 1, H, W)
        """
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        
        output = self.final_conv(dec1)
        output = self.sigmoid(output)
        
        return output


class SceneChangeResNet(nn.Module):
    """
    @param encoder_name: ResNet model adı ('resnet34' veya 'resnet50')
    @param n_classes: Çıkış sınıf sayısı
    """
    def __init__(self, encoder_name='resnet34', n_classes=1):
        super(SceneChangeResNet, self).__init__()
        
        if encoder_name == 'resnet34':
            self.encoder = models.resnet34(pretrained=True)
            encoder_channels = [64, 64, 128, 256, 512]
        elif encoder_name == 'resnet50':
            self.encoder = models.resnet50(pretrained=True)
            encoder_channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Desteklenmeyen encoder: {encoder_name}")
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu),
            nn.Sequential(self.encoder.maxpool, self.encoder.layer1),
            self.encoder.layer2,
            self.encoder.layer3,
            self.encoder.layer4
        ])
        
        self.decoder4 = self._make_decoder_block(encoder_channels[4], encoder_channels[3])
        self.decoder3 = self._make_decoder_block(encoder_channels[3], encoder_channels[2])
        self.decoder2 = self._make_decoder_block(encoder_channels[2], encoder_channels[1])
        self.decoder1 = self._make_decoder_block(encoder_channels[1], encoder_channels[0])
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(encoder_channels[0], 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_classes, 1),
            nn.Sigmoid()
        )
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        features = []
        for layer in self.encoder_layers:
            x = layer(x)
            features.append(x)
        
        # Decoder
        x = self.decoder4(features[4])
        x = self.upsample(x) + features[3]
        
        x = self.decoder3(x)
        x = self.upsample(x) + features[2]
        
        x = self.decoder2(x)
        x = self.upsample(x) + features[1]
        
        x = self.decoder1(x)
        x = self.upsample(x) + features[0]
        
        # Final upsampling to original size
        x = self.upsample(x)
        x = self.final_conv(x)
        
        return x


class CombinedLoss(nn.Module):
    """
    @param bce_weight: BCE loss ağırlığı
    @param dice_weight: Dice loss ağırlığı
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCELoss()
    
    def dice_loss(self, pred, target, smooth=1e-5):
        """
        @param pred: Tahmin edilen maskeler
        @param target: Gerçek maskeler
        @param smooth: Smoothing değeri
        @return: Dice loss değeri
        """
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice_score = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        
        return 1 - dice_score
    
    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        
        total_loss = self.bce_weight * bce + self.dice_weight * dice
        return total_loss, bce, dice


def get_model(model_name='unet', **kwargs):
    """
    @param model_name: Model adı ('unet', 'resnet34', 'resnet50')
    @param kwargs: Model parametreleri
    @return: Model instance
    """
    if model_name == 'unet':
        return SceneChangeUNet(**kwargs)
    elif model_name == 'resnet34':
        return SceneChangeResNet(encoder_name='resnet34', **kwargs)
    elif model_name == 'resnet50':
        return SceneChangeResNet(encoder_name='resnet50', **kwargs)
    else:
        raise ValueError(f"Desteklenmeyen model: {model_name}")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Model test ediliyor...")
    
    test_input = torch.randn(2, 3, 224, 224).to(device)
    test_target = torch.randn(2, 1, 224, 224).to(device)
    
    models_to_test = ['unet', 'resnet34']
    
    for model_name in models_to_test:
        print(f"\n{model_name.upper()} test ediliyor...")
        
        model = get_model(model_name).to(device)
        
        with torch.no_grad():
            output = model(test_input)
            print(f"Input shape: {test_input.shape}")
            print(f"Output shape: {output.shape}")
        
        criterion = CombinedLoss()
        loss, bce, dice = criterion(output, test_target)
        print(f"Total Loss: {loss.item():.4f}, BCE: {bce.item():.4f}, Dice: {dice.item():.4f}")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Toplam parametre: {total_params:,}")
        print(f"Eğitilebilir parametre: {trainable_params:,}")
    
    print("\nModel test tamamlandı!")