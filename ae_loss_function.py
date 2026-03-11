import torch
import torch.nn as nn
from taming.modules.losses.vqperceptual import * 
import numpy as np



class SimpleVAELoss(nn.Module):
    def __init__(self, kl_weight=1e-6, pixel_loss_weight=1.0, 
                 perceptual_weight=1.0, autocorr_weight=1.0,
                 threshold_type='otsu', max_lag=64):
        """
        Enhanced autoencoder loss function with support for automatic threshold binarization.
        Args:
            threshold_type (str): The method for thresholding:
                'fixed': Uses a fixed threshold of 0.5.
                'adaptive': Mean-based adaptive thresholding.
                'otsu': Computes the optimal threshold using the Otsu algorithm.
            max_lag (int): Maximum lag distance (in pixels).
        """
        super().__init__()
        
        self.threshold_type = threshold_type
        self.max_lag = max_lag
        self.kl_weight=kl_weight
        self.pixel_loss_weight=pixel_loss_weight
        self.perceptual_weight=perceptual_weight
        self.autocorr_weight=autocorr_weight
        self.perceptual_loss = None
        if perceptual_weight > 0:
            self.perceptual_loss = LPIPS().eval()
            for param in self.perceptual_loss.parameters():
                param.requires_grad = False
            self._device_cache = None  
        
        
    def scale_to_01(self, img):
        return (img + 1.0) * 0.5

    def compute_threshold(self, scaled_img):
        device = scaled_img.device
        
        if self.threshold_type == 'fixed':
            return torch.tensor(0.5, device=device)
        
        elif self.threshold_type == 'adaptive':
            return scaled_img.mean(dim=(1, 2, 3), keepdim=True)
        
        elif self.threshold_type == 'otsu':
            thresholds = []
            with torch.no_grad():
                for batch in range(scaled_img.shape[0]):
                    img_np = scaled_img[batch, 0].detach().cpu().numpy()
                    
                    # process extreme cases
                    if np.allclose(img_np, img_np[0], atol=1e-5):
                        thresholds.append(0.5)
                        continue
                    
                    hist, bin_edges = np.histogram(
                        img_np, 
                        bins=256, 
                        range=(0, 1),
                        density=True  
                    )
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    
                    # Otsu algorithm
                    total_pixels = img_np.size
                    weight1 = np.cumsum(hist)
                    weight2 = np.cumsum(hist[::-1])[::-1]
                    
                    mean1 = np.cumsum(hist * bin_centers) / (weight1 + 1e-10)
                    mean2 = np.cumsum(hist[::-1] * bin_centers[::-1])[::-1] / (weight2 + 1e-10)
                    
                    variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:])**2
                    
                    if np.all(np.isnan(variance)):
                        thresholds.append(0.5)
                    else:
                        idx = np.nanargmax(variance)
                        thresholds.append(bin_edges[idx])
            
            return torch.tensor(thresholds, device=device).view(-1, 1, 1, 1)
        
        else:
            raise ValueError(f"不支持的阈值类型: {self.threshold_type}")

    def binary_autocorrelation(self, img):
        """
        Calculate two-point autocorrelation map(handles [-1, 1] inputs).
        """
        # transform to range of [0,1]
        scaled_img = self.scale_to_01(img)
        
        threshold = self.compute_threshold(scaled_img)
        
        # transform binary 
        binary_img = (scaled_img.detach() <= threshold).float()
        grad_channel = scaled_img - scaled_img.detach()
        binary_img = binary_img + grad_channel
        
        pad = self.max_lag
        padded_img = nn.functional.pad(binary_img, (pad, pad, pad, pad), mode='constant', value=0)
        
        fft_img = torch.fft.rfft2(padded_img)
        power_spectrum = torch.abs(fft_img)**2
        autocorr = torch.fft.irfft2(power_spectrum, s=padded_img.shape[-2:])
        
        autocorr = autocorr[..., :2*self.max_lag, :2*self.max_lag]
        center = autocorr[..., pad:pad+self.max_lag, pad:pad+self.max_lag]
        
        batch_max = center.amax(dim=(-1,-2), keepdim=True)
        norm_center = center / (batch_max + 1e-8)
        
        return norm_center
    
    @staticmethod
    def radial_weight(size):
        """Create a radially-weighted matrix (higher weights at the center)."""
        y, x = torch.meshgrid(
            torch.arange(size, dtype=torch.float32),
            torch.arange(size, dtype=torch.float32),
            indexing='ij'
        )
        center = size // 2
        r = torch.sqrt((x - center)**2 + (y - center)**2)
        return torch.exp(-r / (size / 4))  
    
    def autocorr_loss(self, inputs, reconstructions):
        """
        Calculate two-point autocorrelation difference loss (handles [-1, 1] inputs).
        """
        input_autocorr = self.binary_autocorrelation(inputs)
        recon_autocorr = self.binary_autocorrelation(reconstructions)
        
        size = input_autocorr.shape[-1]
        weights = self.radial_weight(size).to(inputs.device)
    
        diff = F.smooth_l1_loss(recon_autocorr, input_autocorr, reduction='none', beta=0.1)
        loss = torch.mean(weights * diff)
        
        return loss
        
    
    def forward(self, inputs, reconstructions, posterior, split="train"):
        """
        Calculate total loss
        """
        pixel_loss = torch.abs(inputs - reconstructions).mean()
        
        perceptual_loss = torch.tensor(0.0, device=inputs.device)
        if self.perceptual_weight > 0 and self.perceptual_loss is not None:
            if self._device_cache != inputs.device:
                self.perceptual_loss = self.perceptual_loss.to(inputs.device)
                self._device_cache = inputs.device
            perceptual_loss = self.perceptual_loss(inputs, reconstructions).mean()
        
        kl_loss = posterior.kl().mean()
        autocorr_loss = self.autocorr_loss(inputs, reconstructions)
        
        total_recon_loss = (self.pixel_loss_weight * pixel_loss +
                           self.perceptual_weight * perceptual_loss +
                           self.autocorr_weight * autocorr_loss)
        
        total_loss = total_recon_loss + self.kl_weight * kl_loss
        
        loss_dict = {
            f"{split}_loss/total": total_loss,
            f"{split}_loss/recon": total_recon_loss,
            f"{split}_loss/pixel": pixel_loss,
            f"{split}_loss/perceptual": perceptual_loss,
            f"{split}_loss/autocorr": autocorr_loss,
            f"{split}_loss/kl": kl_loss
        }
        
        return loss_dict