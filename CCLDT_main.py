import torch
torch.set_float32_matmul_precision('high')
import os
os.environ['NCCL_TIMEOUT'] = '3600000'
os.environ['TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC'] = '7200'
##############
import torch.nn as nn
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import os
from DiT import DiT_S_1
from autoencoder_main import AutoencoderKL, DelayedBestModelCheckpoint
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#######
from utils import generate_animation,default, identity
from preprocess_image import train_dataloader,val_dataloader, get_img_shape, unique_compositions
import cv2
from einops import reduce,rearrange
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import all_gather
from PIL import Image, ImageDraw, ImageFont
import textwrap  
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.loggers import TensorBoardLogger


import torch.nn.functional as F
from functools import partial
from collections import namedtuple

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t.long())
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

import math
def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float32)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class DiffusionModel(pl.LightningModule):
    def __init__(self,
                net,
                n_steps: int,
                lr=2e-2,max_epochs=30,scheduler='CosineAnnealingLR',ckpt_path=None,line_lr_decay=0.01,
                continue_line_lr_decay=0.1,continue_lr=None,output_path=None,per_composition_n_sample=10,composition_tensor=None,
                no_cond_batch_n_sample=100,ddim_steps=100,simple_var=False, eta=0, x_T=None,t_start=None,cfg_scale=1,save_interval=5,
                show_training_max_samples=8, show_samples_concat_and_animation=True, autoencoder_path=None,model_id=None,latent_scale_factor=1,
                latent_shape=(4,16,16),
                beta_schedule = 'cosine',
                objective = 'pred_noise',
                min_snr_loss_weight = False, ####With min_snr_loss_weight=False and pred_noise objective, all timesteps are weighted equally at 1.
                min_snr_gamma = 5,
                p_drop=0.1
                ):
        super(DiffusionModel, self).__init__()
        self.net=net
        self.n_steps = n_steps
        
        self.objective = objective
        self.p_drop=p_drop

        assert objective in {'pred_noise', 'pred_x0'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start)'

        self.beta_schedule=beta_schedule
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(self.n_steps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(self.n_steps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        # alphas_bar
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas', alphas.to(torch.float32), persistent=False)
        self.register_buffer('betas', betas.to(torch.float32), persistent=False)
        self.register_buffer('alphas_cumprod', alphas_cumprod.to(torch.float32), persistent=False)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).to(torch.float32),persistent=False)
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod).to(torch.float32), persistent=False)
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1).to(torch.float32), persistent=False)

        snr = alphas_cumprod / (1 - alphas_cumprod)

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            loss_weight = maybe_clipped_snr / snr
        elif objective == 'pred_x0':
            loss_weight = maybe_clipped_snr

        self.register_buffer('loss_weight', loss_weight, persistent=False)

        self.criterion = nn.MSELoss()
        self.lr=lr
        self.max_epochs=max_epochs
        self.scheduler_type=scheduler
        # Settings to continue training from a checkpoint.
        self.continue_lr=continue_lr if continue_lr is not None else lr*line_lr_decay
        self.continue_line_lr_decay=continue_line_lr_decay
        self.ckpt_path=ckpt_path
        self.autoencoder_path=autoencoder_path
        self.last_epoch = 0 
        self.line_lr_decay=line_lr_decay
        # If ckpt_path is provided, load model weights and optimizer state.
        if self.ckpt_path is not None:
            self.load_checkpoint(self.ckpt_path)

        #############Parameters during inference.
        self.ddim_steps=ddim_steps
        self.eta=eta
        self.output_path = output_path
        self.per_composition_n_sample = per_composition_n_sample
        self.composition_tensor=composition_tensor
        self.cfg_scale = cfg_scale
        if composition_tensor is not None:
            self.n_compositions=len(composition_tensor)
            self.no_cond_batch_n_sample=None
            self.with_mix_eps=True
            
        else:
            self.no_cond_batch_n_sample=no_cond_batch_n_sample
            self.n_compositions=no_cond_batch_n_sample//per_composition_n_sample
            self.with_mix_eps=False
        self.show_samples_concat_and_animation=show_samples_concat_and_animation
        self.model_id=model_id
        self.save_img_dir_name=f"gen_imgs/60_1473k_{model_id}_generated_imgs/ddim_steps{ddim_steps}_cfg_scale_{cfg_scale}"
        self.simple_var=simple_var
        self.x_T=x_T
        self.t_start=t_start
        self.batch_n_sample=self.n_compositions*per_composition_n_sample

        ################Parameters for saving comparison plots during training.
        self.save_interval = save_interval  
        self.show_training_max_samples=show_training_max_samples
        self.save_hyperparameters(ignore=['net','composition_tensor'])
        ################
        if self.autoencoder_path is not None:
            self.show_samples_concat_and_animation=False
            self.latent_shape=latent_shape
            self.latent_scale_factor=latent_scale_factor
            self.ae = AutoencoderKL.load_from_checkpoint(autoencoder_path)
            self.ae.eval()
            for param in self.ae.parameters():  
                param.requires_grad = False
    
    # compute x_0 from x_t and pred noise: the reverse of `q_sample`; inverse of Eq.(9) in improved DDPM
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    
    def model_predictions(self, x, t, labels, cfg_scale = None, mask=None):
        if mask is not None:
            model_output = self.net.forward(x, t, labels, mask = mask)
        else:
            assert cfg_scale is not None
            model_output = self.net.forward_with_cfg(x, t, labels, cfg_scale = cfg_scale)
        
        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, model_output)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start_for_pred_noise = x_start 
            pred_noise = self.predict_noise_from_start(x, t, x_start_for_pred_noise)
        
        return ModelPrediction(pred_noise, x_start)
    
    
    def p_losses(self, x_start, t, labels, mask, noise = None,return_xt=False):
        #######x_start is x0
        x_start = x_start.float() if x_start.dtype != torch.float32 else x_start
        noise = default(noise, lambda: torch.randn_like(x_start,dtype=torch.float32))
        # forward diffusion to get x_t
        x = self.sample_forward(x = x_start, t = t, eps = noise)
        model_out=self.net(x, t=t, y=labels, mask=mask)
                      
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start.clone().requires_grad_(True)
        else:
            raise ValueError(f'unknown objective {self.objective}')
        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        # Loss weighting for different timesteps.
        loss = loss * extract(self.loss_weight, t, loss.shape) #if pred noise and do not clamp SNR, the weight is actually not used.
        loss = loss.mean()
        if return_xt:
            return loss, x
        else:
            return loss
    
    def training_step(self, batch, batch_idx):
        (x, c) = batch['image'],batch['composition']
        x = x.to(self.device)
        if self.autoencoder_path is not None:
            self.ae.eval()
            with torch.inference_mode():
                # Training DiTs with random sampling enabled.
                x=self.ae.encode(x).sample()*self.latent_scale_factor
        c = c.to(self.device)
        mask = self.get_masked_context(context=c, p=self.p_drop)
        
        #print(c[0])
        current_batch_size = x.shape[0]
        # Timestep t starts from 0, and the timesteps for each sample are independent.
        t = torch.randint(0, self.n_steps, (current_batch_size, ),dtype=torch.int64).to(self.device)
        assert x.device == t.device == self.alphas_cumprod.device, \
            f"设备不匹配: x={x.device}, t={t.device}, alpha_bar={self.alphas_cumprod.device}"
        loss, x_t = self.p_losses(x, t, c, mask, noise = None, return_xt=True)
        self.log("train_loss", loss,on_epoch=True,on_step=False,sync_dist=True,batch_size=current_batch_size)
        lr = self.optim.param_groups[0]['lr']
        self.log('learning_rate', lr, on_epoch=True,on_step=False,sync_dist=True,batch_size=current_batch_size)  # Saving to TensorBoard
        ####################
    
        if batch_idx == len(self.trainer.train_dataloader) - 1:
            self.show_training_max_samples = min(x.shape[0], self.show_training_max_samples)
            
            x_subset = x[:self.show_training_max_samples] if self.autoencoder_path is None else batch["image"][:self.show_training_max_samples]
            x_t_subset = x_t[:self.show_training_max_samples]
            c_subset = c[:self.show_training_max_samples] if c is not None else None
            mask_subset = mask[:self.show_training_max_samples] if mask is not None else None
            t_subset = t[:self.show_training_max_samples] if t is not None else None
            self.last_batch = (
                x_subset,
                x_t_subset,
                #x_denoised_subset.detach().cpu(),
                c_subset if c_subset is not None else None,
                mask_subset if mask_subset is not None else None,
                t_subset if t_subset is not None else None
            )

        else:
            self.last_batch = None
         
        ###################
        return loss
    
    ############
    def on_train_epoch_end(self):
        is_last_epoch = (self.current_epoch + 1) == self.trainer.max_epochs
        is_interval_epoch = ((self.current_epoch+1) % self.save_interval) == 0
        
        if (is_interval_epoch or is_last_epoch): #and self.trainer.is_global_zero:
            if hasattr(self, 'last_batch') and self.last_batch is not None:
                x, x_t, c, mask, t = self.last_batch
                x_denoised = self.denoised(
                    x_T=x_t,
                    t_start=t,
                    context=c,
                    mask=mask,
                    simple_var=self.simple_var,
                    inference_transform=lambda x: (x + 1) / 2,
                    with_mix_eps=False,
                )

                
                x_cpu = x.detach().cpu()
                if self.autoencoder_path is not None:
                    x_t_cpu=None
                else:
                    x_t_cpu=x_t.detach().cpu()
                x_denoised_cpu=x_denoised.detach().cpu()
                c_cpu=c.detach().cpu() 
                mask_cpu=mask.detach().cpu()
                t_cpu=t.detach().cpu()
            
                self.save_tensor_images(
                    x_cpu, x_denoised_cpu,
                    cur_epoch=self.current_epoch,
                    file_dir=self.output_path,
                    x_noised=x_t_cpu,
                    save_dir = os.path.join(self.output_path, f"saved-images/diff_model/{self.model_id}"),
                    font_size=24,
                    column_padding=10,
                    row_spacing=10,
                    #max_samples=self.show_training_max_samples,
                    cond=c_cpu,
                    mask=mask_cpu,
                    t=t_cpu,
                    max_samples_per_image=5
                )
                # del self.last_batch
                self.last_batch=None

    ##########################
    def on_validation_start(self):
        self.val_batches_per_epoch = len(self.trainer.val_dataloaders)


    def validation_step(self, batch, batch_idx):
        self.net.eval()
        with torch.inference_mode():
            (x, c) = batch['image'], batch['composition']
            x = x.to(self.device)
            if self.autoencoder_path is not None:
                self.ae.eval()
                x=self.ae.encode(x).sample()*self.latent_scale_factor
            c = c.to(self.device)
            current_batch_size = x.shape[0]
            # Do not use masking during validation; retain all conditional information.
            mask = self.get_masked_context(context=c, p=0)
            t = torch.randint(0, self.n_steps, (current_batch_size,),device=self.device, dtype=torch.int64)
            val_loss, x_t = self.p_losses(x, t, c, mask, noise = None, return_xt=True)
            self.log("val_loss", val_loss, on_epoch=True,on_step=False,sync_dist=True,batch_size=current_batch_size)
            
            is_last_epoch = (self.current_epoch + 1) == self.trainer.max_epochs
            is_interval_epoch = ((self.current_epoch+1) % self.save_interval) == 0
            if (is_interval_epoch or is_last_epoch):
                ############Sampling
                if self.current_epoch>=self.max_epochs//2:
                    x_T = torch.randn_like(x, dtype=torch.float32, device=self.device)
                    ###########
                    cal_fid_sample_dir = f"{self.output_path}/saved-images/diff_model/{self.model_id}/validation-sample/{self.current_epoch}"
                    self.sample(x_T=x_T,com=c,mask=mask,batch_id= batch_idx,refer_batch_size=current_batch_size,
                                simple_var=self.simple_var,with_mix_eps=False,cfg_scale=None,save_dir=cal_fid_sample_dir)
                ##############
                if batch_idx == self.val_batches_per_epoch - 1:
                    self.show_valid_max_samples = min(x.shape[0], self.show_training_max_samples)
                    x_denoised = self.denoised(
                            x_T=x_t,
                            t_start=t,
                            context=c,
                            mask=mask,
                            simple_var=self.simple_var,
                            inference_transform=lambda x: (x + 1) / 2,
                            with_mix_eps=False,
                            cfg_scale=None
                        )
                    self.last_val_batch = (batch['image'][:self.show_valid_max_samples], x_t[:self.show_valid_max_samples],
                                            c[:self.show_valid_max_samples], mask[:self.show_valid_max_samples],
                                            t[:self.show_valid_max_samples], x_denoised[:self.show_valid_max_samples])
            return val_loss
    
    ######################
    def on_validation_epoch_end(self):
        is_last_epoch = (self.current_epoch + 1) == self.trainer.max_epochs
        is_interval_epoch = ((self.current_epoch+1) % self.save_interval) == 0
        if (is_interval_epoch or is_last_epoch):
            if hasattr(self, 'last_val_batch'):
                x, x_t, c, mask, t, x_denoised = self.last_val_batch
                x_cpu = x.detach().cpu()
                if self.autoencoder_path is not None:
                    x_t_cpu=None
                else:
                    x_t_cpu=x_t.detach().cpu()
                x_denoised_cpu=x_denoised.detach().cpu()
                c_cpu=c.detach().cpu() 
                mask_cpu=mask.detach().cpu()
                t_cpu=t.detach().cpu()
        
                self.save_tensor_images(
                    x_cpu, x_denoised_cpu,
                    cur_epoch=self.current_epoch,
                    file_dir=self.output_path,
                    x_noised=x_t_cpu,
                    save_dir = os.path.join(self.output_path, f"saved-images/diff_model/{self.model_id}"),
                    font_size=24,
                    column_padding=10,
                    row_spacing=10,
                    #max_samples=self.show_training_max_samples,
                    cond=c_cpu,
                    mask=mask_cpu,
                    t=t_cpu,
                    descri="valid",
                    max_samples_per_image=5
                )
                del self.last_val_batch
            torch.cuda.empty_cache()
        
   
    def configure_optimizers(self):
       
        self.optim = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999), 
            weight_decay=0,
        )
       
        self.gradient_clip_val = 1.0  
        
        if self.ckpt_path is not None:
            self.scheduler_type='linear'
            checkpoint = torch.load(self.ckpt_path, map_location="cpu")
            if "optimizer_states" in checkpoint:
                self.optim.load_state_dict(checkpoint["optimizer_states"][0])
                for param_group in self.optim.param_groups:
                    param_group["lr"] = self.continue_lr 
                print(f"从 {self.ckpt_path} 加载优化器状态，并让学习率调度器从 epoch {self.last_epoch} 继续")
       
        if self.scheduler_type=='linear':
            scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optim,
                start_factor=1,                      
                end_factor=self.line_lr_decay if self.ckpt_path is None else self.continue_line_lr_decay,                      
                total_iters=self.trainer.estimated_stepping_batches,
                last_epoch= -1
            )
            updata_type='step'
        
        else:
            raise ValueError("Please choose one of [CosineAnnealingLR, linear, cos_warm_decay,cos_warmrestart]")   
        
        scheduler.base_lrs = [param_group["lr"] for param_group in self.optim.param_groups]
        return {
                "optimizer": self.optim,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": updata_type  
                }
            }
    
    def load_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        # load model weights
        self.net.load_state_dict(checkpoint['model_state_dict'])
        if 'epoch' in checkpoint:
            self.last_epoch = checkpoint['epoch']+1 
            


    #################
    
    def forward(self, *args, **kwargs):
        """
        占位的 forward 方法。
        """
        raise NotImplementedError("This model does not use forward for training or inference.")



    def sample_forward(self, x, t, eps=None):
        alpha_bar = self.alphas_cumprod[t].reshape(-1, 1, 1, 1).to(x.device)
        #alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        if eps is None:
            eps = torch.randn_like(x)
        res = eps.to(x.device) * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * x
        return res.float()

    def ddpm_sample_backward(self, x_T, context=None,mask=None,simple_var=True,t_start=None,save_rate=20,inference_transform=lambda x: (x+1)/2, with_mix_eps=True, cfg_scale=1):
        with torch.no_grad(): 
            x = x_T.clone() 
            if t_start is None:
                t_start = self.n_steps-1
            if self.autoencoder_path is None:
                intermediate_samples = [x.detach().cpu()] # samples at T = timesteps
                t_steps = [t_start] # keep record of time to use in animation generation
            # Reverse process: If t_start is not specified, t should run from 999 down to 0 (inclusive). Note that t=0 corresponds to x_1, not the original image x_0.
            for t in tqdm(range(t_start, -1, -1), 'DDPM sampling'):
                x = self.sample_backward_step(x, t, simple_var,context=context,mask=mask,with_mix_eps=with_mix_eps, cfg_scale=cfg_scale)
                if self.autoencoder_path is None:
                    if t % save_rate == 0 or t < 10:
                        intermediate_samples.append(inference_transform(x.detach().cpu()))
                        t_steps.append(t-1)

                #########
            if self.autoencoder_path is None:
                return inference_transform(x.detach().cpu()),intermediate_samples, t_steps
            else:
                x=self.ae.decode(x/self.latent_scale_factor)
                return inference_transform(x.detach().cpu())

    #################################DDIM implementary
    def ddim_sample_backward(self, x_T, context=None,mask=None,simple_var=False,t_start=None,inference_transform=lambda x: (x+1)/2,
                        with_mix_eps=True, cfg_scale=1, 
                        ddim_step=20, eta=0
                        ):
        with torch.no_grad():
            x = x_T.clone()  
            if simple_var:
                # var/sigma^2=beta_t
                eta = 1
            # Defaults to n_steps if t_start is not specified.
            if t_start is None:
                t_start = self.n_steps
            # For example, if self.n_steps=100 and ddim_step=10, the ts array contains [100, 90, ..., 0].
            ts = torch.linspace(t_start, 0,
                                (ddim_step + 1)).to(x.device).to(torch.long)
            intermediate_samples = [x.detach().cpu()] # samples at T = timesteps
            t_steps = [t_start] # keep record of time to use in animation generation
            for i in tqdm(range(1, ddim_step + 1),
                        f'DDIM sampling with eta {eta} simple_var {simple_var}'):
                # larger timestep t2, beta ranges from t=0 up to n_steps-1,so we need to subtract 1 here.
                cur_t = ts[i - 1] - 1
                # smaller timestep t1
                prev_t = ts[i] - 1
                ab_cur = self.alphas_cumprod[cur_t]
                ab_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0, dtype=self.alphas_cumprod.dtype, device=self.alphas_cumprod.device)

                if isinstance(cur_t, torch.Tensor):
                    t_tensor = cur_t.clone().detach().to(self.device)
                else:
                    t_tensor = torch.tensor(cur_t, dtype=torch.float, device=self.device)

                t_batch = t_tensor.expand(x.shape[0])  
                ############
                if with_mix_eps:
                    if cfg_scale is None:
                        raise ValueError(
                        "`cfg_scale` must be provided when using CFG guidence conditional generation. "
                    )
                    else:
                        pred_noise, x_start, *_ = self.model_predictions(x, t_batch, context, cfg_scale = cfg_scale, mask=None)
                else:
                    pred_noise, x_start, *_ = self.model_predictions(x, t_batch, context, cfg_scale = None, mask = mask)


                ###########
                var = eta**2 * (1 - ab_prev) / (1 - ab_cur) * (1 - ab_cur / ab_prev)
                noise = torch.randn_like(x)
                
                if simple_var:
                    third_term = (1 - ab_cur / ab_prev)**0.5 * noise
                else:
                    third_term = var**0.5 * noise
               
                c = (1 - ab_prev - var).sqrt()
                x = x_start * ab_prev.sqrt() + \
                  c * pred_noise + \
                  third_term

                intermediate_samples.append(inference_transform(x.detach().cpu()))
                t_steps.append(prev_t)
            if self.autoencoder_path is None:
                return inference_transform(x.detach().cpu()),intermediate_samples, t_steps
            else:
                x=self.ae.decode(x/self.latent_scale_factor)
                return inference_transform(x.detach().cpu())
    #######################################
   
    def denoised(self, x_T, context=None, mask=None, simple_var=True, t_start=None, inference_transform=lambda x: (x+1)/2,with_mix_eps=True, cfg_scale=1):
        with torch.no_grad():  
            x = x_T.clone()
            batch_size = x.shape[0]
            device = x.device
            if t_start is None:
                t_start = torch.full((batch_size,), self.n_steps - 1, dtype=torch.long, device=device)
            else:
                t_start = t_start.to(device).long()

            max_t = t_start.max().item()
            for current_t in range(max_t, -1, -1):
                active_mask = (t_start >= current_t) & (current_t >= 0)
                if not active_mask.any():
                    continue
                
                x_subset = x[active_mask]
                context_subset = context[active_mask] if context is not None else None
                mask_subset = mask[active_mask] if mask is not None else None
    
                x_prev = self.sample_backward_step(x_subset, current_t, simple_var, context=context_subset, mask=mask_subset,with_mix_eps=with_mix_eps,cfg_scale=cfg_scale)
                
                x[active_mask] = x_prev
                del x_prev, x_subset, context_subset, mask_subset
                torch.cuda.empty_cache() 
            if self.autoencoder_path is None:
                return inference_transform(x.detach().cpu())
            else:
                x=self.ae.decode(x/self.latent_scale_factor)
                return inference_transform(x.detach().cpu())
        ################################

    def sample_backward_step(self, x_t, t, simple_var=True,context=None,mask=None, with_mix_eps=True, cfg_scale=1):
        
        if isinstance(t, torch.Tensor):
            t_tensor = t.clone().detach().to(self.device)
        else:
            t_tensor = torch.tensor(t, dtype=torch.float, device=self.device)
        t_batch = t_tensor.expand(x_t.shape[0])  
        if with_mix_eps:
            if cfg_scale is None:
                raise ValueError(
                "`cfg_scale` must be provided when using CFG guidence conditional generation. "
            )
            else:
                pred_noise, x_start, *_ = self.model_predictions(x_t, t_batch, context, cfg_scale = cfg_scale, mask=None)
        else:
            pred_noise, x_start, *_ = self.model_predictions(x_t, t_batch, context, cfg_scale = None, mask=mask)

        if t == 0:
            noise = 0
        else:
            if simple_var:
                var = self.betas[t]
            else:
                var = (1 - self.alphas_cumprod[t - 1]) / (
                    1 - self.alphas_cumprod[t]) * self.betas[t]
            noise = torch.randn_like(x_t).to(self.device)
            noise *= torch.sqrt(var.to(self.device))
        mean = (x_t -
                (1 - self.alphas[t]) / torch.sqrt(1 - self.alphas_cumprod[t]) *
                pred_noise) / torch.sqrt(self.alphas[t])
        x_t = mean + noise

        return x_t


    def sample(self, x_T, com, mask, batch_id, refer_batch_size, simple_var=False, 
               with_mix_eps=True, cfg_scale=5,save_dir=None):
        
        save_dir = f"{self.output_path}/{self.save_img_dir_name}" if save_dir is None else save_dir
        os.makedirs(save_dir, exist_ok=True)
        rank = getattr(self.trainer, "global_rank", 0)
        with torch.no_grad(): 
            x = x_T.clone() 
            sample_outputs =self.ddim_sample_backward(x_T=x, context=com,mask=mask,simple_var=simple_var,t_start=None,inference_transform=lambda x: (x+1)/2,
                        with_mix_eps=with_mix_eps, cfg_scale=cfg_scale, 
                        ddim_step=self.ddim_steps, eta=self.eta)
            if self.autoencoder_path is None:
                x = sample_outputs[0]
            else:
                x = sample_outputs
            ###########
            print(f"生成图像像素范围: [{x.min():.3f}, {x.max():.3f}]")  # should be [0,1]
           
            for idx in range(x.shape[0]):
                comp_info = com[idx].cpu().numpy() 
                comp_str = "_".join([f"{v:.3f}" for v in comp_info])  
                
                per_gpu_sample_id=batch_id*refer_batch_size+idx
                filename = f"comp_{comp_str}_gpu{rank}_cfgscale{cfg_scale}_sample{per_gpu_sample_id}.png"
                filepath = os.path.join(save_dir, filename)
                img_array = (x[idx].squeeze() * 255).clamp(0, 255).numpy().astype(np.uint8)
                Image.fromarray(img_array).save(filepath)
            
            return x


    
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
            #print(f"Running predict_step, batch_idx: {batch_idx}")
            self.net.eval()
            self.net.to(self.device)  
            with torch.no_grad():
                if self.show_samples_concat_and_animation and self.autoencoder_path is None:
                    (x_T,c,mask)=batch
                    x_T=x_T.to(self.device)
                    c=c.to(self.device)
                    mask=mask.to(self.device)
                    imgs,intermediate_samples, t_steps =self.ddim_sample_backward(x_T=x_T, context=c,mask=mask,simple_var=self.simple_var,t_start=self.t_start,inference_transform=lambda x: (x+1)/2,
                        with_mix_eps=self.with_mix_eps, cfg_scale=self.cfg_scale, 
                        # The eta ranges from 0 to 1, eta=1 corresponds to the DDPM formulation, which yields better performance when paired with 'simple_var'. Conversely, for DDIM sampling, eta=0achieves optimal results without the use of 'simple_var'.
                        ddim_step=self.ddim_steps, eta=self.eta) 
                    
                   
                    imgs = imgs.to(self.device)
                    gathered_imgs = [torch.zeros_like(imgs) for _ in range(self.trainer.world_size)] 
                    all_gather(gathered_imgs, imgs)  
                    all_imgs = torch.cat(gathered_imgs, dim=0).cpu()  
                    all_imgs = all_imgs.view(self.trainer.world_size, 2, self.n_compositions, self.per_composition_n_sample//self.trainer.world_size,*get_img_shape())
                    batch_0_imgs = torch.cat([all_imgs[i, 0] for i in range(self.trainer.world_size)], dim=1)
                    batch_1_imgs = torch.cat([all_imgs[i, 1] for i in range(self.trainer.world_size)], dim=1)
                    batch_0_imgs = batch_0_imgs.view(self.n_compositions * self.per_composition_n_sample, *batch_0_imgs.shape[-3:])
                    batch_1_imgs = batch_1_imgs.view(self.n_compositions * self.per_composition_n_sample, *batch_1_imgs.shape[-3:])
                    
                    
                    # process intermediate_samples
                    batch_0_intermediate_samples=[[] for _ in range(len(intermediate_samples))]
                    batch_1_intermediate_samples=[[] for _ in range(len(intermediate_samples))]
                    for i, sample in enumerate(intermediate_samples):
                        sample = sample.to(self.device)  
                        gathered_sample = [torch.zeros_like(sample, device=self.device) for _ in range(self.trainer.world_size)]
                        all_gather(gathered_sample, sample)  
                        gathered_intermediate_samples = torch.cat(gathered_sample, dim=0).cpu()  
                        gathered_intermediate_samples = gathered_intermediate_samples.view(self.trainer.world_size, 2, self.n_compositions, self.per_composition_n_sample//self.trainer.world_size, *get_img_shape())
                        batch_0_intermediate_samples[i] = torch.cat([gathered_intermediate_samples[j, 0] for j in range(self.trainer.world_size)], dim=1)
                        batch_1_intermediate_samples[i] = torch.cat([gathered_intermediate_samples[j, 1] for j in range(self.trainer.world_size)], dim=1)
                        batch_0_intermediate_samples[i] = batch_0_intermediate_samples[i].view(self.n_compositions * self.per_composition_n_sample, *get_img_shape())
                        batch_1_intermediate_samples[i] = batch_1_intermediate_samples[i].view(self.n_compositions * self.per_composition_n_sample, *get_img_shape())

                    global_batch_idx = batch_idx + self.trainer.global_rank * (2 // self.trainer.world_size)
                    self.create_dirs(self.output_path)   
                    if self.trainer.global_rank == 0:
                        # 保存图片
                        save_path_0 = f"{self.output_path}/saved-images/diff_model/{self.model_id}/no_composition_batch_{global_batch_idx * 2}.png"
                        save_path_1 = f"{self.output_path}/saved-images/diff_model/{self.model_id}/no_composition_batch_{global_batch_idx * 2 + 1}.png"

                        save_image(batch_0_imgs, save_path_0, nrow=self.per_composition_n_sample)
                        save_image(batch_1_imgs, save_path_1, nrow=self.per_composition_n_sample)

                        print(f"Saved batch {global_batch_idx * 2} images at {save_path_0}")
                        print(f"Saved batch {global_batch_idx * 2 + 1} images at {save_path_1}")
                        # 保存中间去噪过程gif
                        save_inetrmediate_path_0 = f"{self.output_path}/generated-images/{self.model_id}/no_composition_batch_{global_batch_idx * 2}.gif"
                        save_intermediate_path_1 = f"{self.output_path}/generated-images/{self.model_id}/no_composition_batch_{global_batch_idx * 2 + 1}.gif"
                        generate_animation(
                            batch_0_intermediate_samples,
                            t_steps, 
                            save_inetrmediate_path_0,
                            self.per_composition_n_sample
                            )
                        generate_animation(
                            batch_1_intermediate_samples,
                            t_steps, 
                            save_intermediate_path_1,
                            self.per_composition_n_sample
                            )

                        print(f"Saved batch {global_batch_idx * 2} images gif at {save_inetrmediate_path_0}")
                        print(f"Saved batch {global_batch_idx * 2 + 1} images gif at {save_intermediate_path_1}")

                else:
                    (x_T, c, mask)=batch
                    x_T=x_T.to(self.device)
                    c=c.to(self.device)
                    mask=mask.to(self.device)
                    self.sample(x_T, c, mask, batch_id=batch_idx, refer_batch_size=self.batch_n_sample, simple_var=self.simple_var, with_mix_eps=self.with_mix_eps, cfg_scale=self.cfg_scale)
            


    def predict_dataloader(self):
        # For visualization purposes only.
        if self.show_samples_concat_and_animation:
            dataset = XTDataset(
                x_T=self.x_T,  
                num_batch=2,
                per_composition_n_sample=self.per_composition_n_sample,
                composition_tensor=self.composition_tensor if self.composition_tensor is not None else None,
                img_shape=get_img_shape() if self.autoencoder_path is None else self.latent_shape,
                no_cond_batch_n_sample=self.no_cond_batch_n_sample,
                device='cpu'  
            )
        else:
            dataset = XTDataset(
            x_T=self.x_T,  
            num_batch=1,
            per_composition_n_sample=self.per_composition_n_sample,
            composition_tensor = self.composition_tensor,
            img_shape=get_img_shape() if self.autoencoder_path is None else self.latent_shape,
            no_cond_batch_n_sample= self.no_cond_batch_n_sample,
            device='cpu'  
            )
            
            self.batch_n_sample=self.per_composition_n_sample
        
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.trainer.world_size,  
            rank=self.trainer.global_rank,
            shuffle=False
        )
        return torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=self.batch_n_sample if self.batch_n_sample<=400 else 400, shuffle=False,num_workers=4,pin_memory=True)  
    
    #########
    def create_dirs(self, file_dir):
        dir_names = ["generated-images", "saved-images"]
        for dir_name in dir_names:
            os.makedirs(os.path.join(file_dir, dir_name), exist_ok=True)

    
    def save_tensor_images(
        self,
        x_orig,
        x_denoised,
        cur_epoch,
        file_dir,
        x_noised=None,
        save_dir=None,
        cond=None,
        mask=None,
        t=None,
        font_path="arial.ttf",
        font_size=24,
        column_padding=10,
        row_spacing=15,
        descri="train",
        max_samples_per_image=4  # Maximum samples per image (default: 4).
        ):
        """Save image comparison plots (supports saving across multiple files)."""
        
        if save_dir is None:
            save_dir = os.path.join(file_dir, f"saved-images/diff_model/{self.model_id}")
        os.makedirs(save_dir, exist_ok=True)

        has_noised = x_noised is not None
        suffix = f"{descri}_x_orig_noised_denoised" if has_noised else f"{descri}_x_orig_denoised"

        num_samples = len(x_orig)
        num_images = (num_samples + max_samples_per_image - 1) // max_samples_per_image

        for img_index in range(num_images):
            start_idx = img_index * max_samples_per_image
            end_idx = min((img_index + 1) * max_samples_per_image, num_samples)
           
            x_orig_sub = x_orig[start_idx:end_idx]
            x_denoised_sub = x_denoised[start_idx:end_idx]
            x_noised_sub = x_noised[start_idx:end_idx] if x_noised is not None else None
            cond_sub = cond[start_idx:end_idx] if cond is not None else None
            mask_sub = mask[start_idx:end_idx] if mask is not None else None
            t_sub = t[start_idx:end_idx] if t is not None else None

            fpath = os.path.join(save_dir, f"{suffix}_{cur_epoch}_part{img_index+1}_gpu{self.global_rank}.jpeg")
            self._build_single_image(
                x_orig_sub, x_denoised_sub, x_noised_sub, cond_sub, mask_sub, t_sub,
                fpath, font_path, font_size, column_padding, row_spacing)

    def _build_single_image(
        self,
        x_orig,
        x_denoised,
        x_noised,
        cond,
        mask,
        t,
        fpath,
        font_path,
        font_size,
        column_padding,
        row_spacing,
        ):
        """Construct and save a single image (internal helper function)."""
        has_noised = x_noised is not None
        # tansform to  [0,1]
        inference_transform = lambda x: (x + 1) / 2
        
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default(size=font_size)
                print("使用默认字体")

        
        def create_image_with_text(img, text, padding=column_padding):
            padded_img = Image.new("L", (img.width + 2*padding, img.height + 2*padding), 255)
            padded_img.paste(img, (padding, padding))
            
            text_img = Image.new("L", (padded_img.width, font_size + 20), 255)
            draw = ImageDraw.Draw(text_img)
            
            max_text_width = padded_img.width - 20
            if font.getlength(text) > max_text_width:
                truncated_text = text
                while font.getlength(truncated_text + "...") > max_text_width and len(truncated_text) > 3:
                    truncated_text = truncated_text[:-1]
                text = truncated_text + "..."
            
            text_width = font.getlength(text)
            x = (text_img.width - text_width) // 2
            draw.text((x, 10), text, fill=0, font=font)
            
            combined = Image.new("L", (padded_img.width, padded_img.height + text_img.height))
            combined.paste(padded_img, (0, 0))
            combined.paste(text_img, (0, padded_img.height))
            return combined

        sample_blocks = []
        for i in range(len(x_orig)):
            # Original images
            img_orig = Image.fromarray(
                (inference_transform(x_orig[i]).squeeze().cpu().numpy() * 255).astype(np.uint8),
                mode='L'
            )
            
            # Denoised images
            img_denoised = x_denoised[i].cpu().numpy().squeeze(0)
            img_denoised = (img_denoised * 255).astype(np.uint8)
            img_denoised = Image.fromarray(img_denoised, mode='L')
            
            # prepare texts
            orig_text = "Original"
            denoised_text = "Denoised"
            
            if cond is not None:
                cond_str = " | ".join([f"{x:.3f}" for x in cond[i].cpu().numpy().flatten()])
                orig_text = f"{cond_str}"
                
                if mask is not None:
                    cond_mask= mask[i].item()
                    denoised_text = f"Mask={cond_mask}"
            
            if t is not None:
                denoised_text += f" | t={int(t[i])}"
            
            orig_block = create_image_with_text(img_orig, orig_text)
            denoised_block = create_image_with_text(img_denoised, denoised_text)
            
            if has_noised:
                img_noised = Image.fromarray(
                    (inference_transform(x_noised[i]).squeeze().cpu().numpy() * 255).astype(np.uint8),
                    mode='L'
                )
                noised_text = "Noised"
                if t is not None:
                    noised_text += f" | t={int(t[i])}"
                noised_block = create_image_with_text(img_noised, noised_text)
                
                sample_width = max(orig_block.width, noised_block.width, denoised_block.width)
                sample_height = orig_block.height + noised_block.height + denoised_block.height + 2*row_spacing
                sample_img = Image.new("L", (sample_width, sample_height), 255)
                
                sample_img.paste(orig_block, ((sample_width - orig_block.width) // 2, 0))
                sample_img.paste(noised_block, ((sample_width - noised_block.width) // 2, orig_block.height + row_spacing))
                sample_img.paste(denoised_block, ((sample_width - denoised_block.width) // 2, orig_block.height + noised_block.height + 2*row_spacing))
            else:
                sample_width = max(orig_block.width, denoised_block.width)
                sample_height = orig_block.height + denoised_block.height + row_spacing
                sample_img = Image.new("L", (sample_width, sample_height), 255)
                sample_img.paste(orig_block, ((sample_width - orig_block.width) // 2, 0))
                sample_img.paste(denoised_block, ((sample_width - denoised_block.width) // 2, orig_block.height + row_spacing))
            
            sample_blocks.append(sample_img)
        
        total_width = sum(block.width for block in sample_blocks) + column_padding * (len(sample_blocks) - 1)
        total_height = max(block.height for block in sample_blocks)
        
        combined = Image.new("L", (total_width, total_height), 255)
        x_offset = 0
        for block in sample_blocks:
            combined.paste(block, (x_offset, (total_height - block.height) // 2))
            x_offset += block.width + column_padding
       
        combined.save(fpath)
        print(f"保存图像到: {fpath}")



    def get_masked_context(self, context, p=0.1):
        "Randomly mask out context"
        """随机掩盖成分信息"""
        num_samples=len(context)
        mask = torch.rand((num_samples, 1), device=context.device) < p
        return mask
    


class XTDataset(Dataset):
    def __init__(self, x_T=None, num_batch=1,per_composition_n_sample=10,img_shape=get_img_shape(),composition_tensor=None,no_cond_batch_n_sample=None,device='cuda'):
        super().__init__()
        # `x_T_values` 是传入的预定义的 x_T 张量
        self.device = device
        self.num_batch=num_batch
        if composition_tensor is not None:
            if not isinstance(composition_tensor, torch.Tensor):
                raise TypeError("composition_tensor 必须为 torch.Tensor 类型")
            if composition_tensor.ndim != 2 or composition_tensor.shape[1] != 3:
                raise ValueError("composition_tensor 形状应为 (num_compositions, 3)")

            self.composition_tensor = composition_tensor.to(dtype=torch.float32, device="cpu")
            num_compositions = self.composition_tensor.shape[0]
            self.batch_n_sample = num_compositions * per_composition_n_sample
            cond = self.composition_tensor.repeat_interleave(per_composition_n_sample, dim=0)
            self.cond = cond.repeat(num_batch, 1)  
            self.mask=torch.zeros((self.cond.size(0), 1), dtype=torch.bool, device=self.cond.device)
        else:
            # 无条件生成逻辑
            if no_cond_batch_n_sample is None:
                raise ValueError("无条件生成时需指定 no_cond_batch_n_sample")
            self.batch_n_sample = no_cond_batch_n_sample
            self.cond = torch.zeros(
                (num_batch * self.batch_n_sample,3),  dtype=torch.float32, device="cpu")
            self.mask = torch.ones((self.cond.size(0), 1), dtype=torch.bool, device=self.cond.device)
                
        #############生成 x_T ################
        self.total_samples = num_batch * self.batch_n_sample  # 总样本数
        self.shape = (self.total_samples, *img_shape)
        # 数据的初始化放在cpu上
        if x_T is not None:
            if x_T.shape != self.shape:
                raise ValueError(f"x_T 形状应为 {self.shape}, 但传入的是 {x_T.shape}")
            self.x_T = torch.tensor(x_T, dtype=torch.float32).cpu()
        else:
            self.x_T = torch.randn(self.shape, dtype=torch.float32, device='cpu')


    def __len__(self):
        return self.total_samples  
    def __getitem__(self, idx):
        return self.x_T[idx], self.cond[idx],self.mask[idx]
        


if __name__ == '__main__':
    max_epochs=1000
    save_interval=100
    n_steps = 1000
    default_root_dir="PhaseSimuDiffusionModel/time60_cond_ddpm/DITs" 
    autoencoder_path="PhaseSimuDiffusionModel/time60_cond_ddpm/DITs/saved-models/AE/AE_v9_2/last.ckpt"
    model_id=f"v14_mlp_t+sa_com+block_FFN+pred_x0_epoch_{max_epochs}"
    DIT_config={
    "input_size":32,
    "in_channels":4,
    "t_embed_dim":256,
    "t_use_pos_encoding":False,
    "y_embed_dim":256,
    "learn_sigma":False,
    "com_encoder":'SAComEncoderCLS',         ###Options: 'RepeatComEncoder' or 'SAComEncoderCLS'
    "DiT_block_shared_FFN":False}
    latent_scale_factor=0.997318
    latent_shape=(4,32,32)
    net=DiT_S_1(**DIT_config)
    lr=1e-04
    scheduler='linear'
    line_lr_decay=0.1
    ##########
    beta_schedule = 'linear'
    objective = 'pred_noise'
    # objective='pred_x0'
    # min_snr_loss_weight = False
    min_snr_loss_weight = True
    min_snr_gamma = 5
    p_drop=0.1
    
    
    ##################################### Training
    ckpt_path=None
    model = DiffusionModel(net=net,n_steps=n_steps,max_epochs=max_epochs,lr=lr,scheduler=scheduler,ckpt_path=ckpt_path,line_lr_decay=line_lr_decay,continue_line_lr_decay=0.1,
                        continue_lr=4e-4,output_path=default_root_dir,cfg_scale=1,save_interval=save_interval,show_training_max_samples=5,  #####show_training_max_samples: represents the number of samples per GPU selected for visualization during training
                        ddim_steps=100,simple_var=False, eta=0,autoencoder_path=autoencoder_path,model_id=model_id,latent_scale_factor=latent_scale_factor,
                        latent_shape=latent_shape,
                        beta_schedule = beta_schedule,
                        objective = objective,
                        min_snr_loss_weight = min_snr_loss_weight,
                        min_snr_gamma = min_snr_gamma,
                        p_drop=p_drop)
    save_dir = os.path.join(default_root_dir, f"saved-models/diff_model/{model_id}")
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_callback = DelayedBestModelCheckpoint(
            start_epoch_ratio=0.8,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="best-{epoch:02d}",
            dirpath=save_dir,  
            save_last=False,
            every_n_epochs=1,
            verbose=True
        )
    
    last_model_callback = ModelCheckpoint(
        save_last=True,  
        filename="last-{epoch}",
        dirpath=save_dir
    )
    
    logger = TensorBoardLogger(
    save_dir="PhaseSimuDiffusionModel/time60_cond_ddpm/DITs/lightning_logs",  # Log storage directory
    name="Diffu", 
    # version=1  # Optional: Specify version number
    )
    
    trainer = pl.Trainer(max_epochs=max_epochs, devices=1, accelerator="gpu",strategy='ddp_find_unused_parameters_false',default_root_dir=default_root_dir,
                            gradient_clip_val=1.0,
                            gradient_clip_algorithm="norm",
                            callbacks=[checkpoint_callback, last_model_callback],
                            # callbacks=[last_model_callback],
                            logger=logger,)
    trainer.fit(model, train_dataloader,val_dataloaders=val_dataloader)
   
    ###################################### Save model
    checkpoint = {
        'epoch': model.current_epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_states':model.optim.state_dict()
    }
    net_path = os.path.join(save_dir, f"{model_id}.pth")
    torch.save(checkpoint, net_path)
    
    
    
    ######Reference/Sampling################
    ckpt_path=f"PhaseSimuDiffusionModel/time60_cond_ddpm/DITs/saved-models/diff_model/{model_id}/{model_id}.pth"
    composition_tensor=unique_compositions
    cfg_scale=2.5
    ddim_steps=100
    per_composition_n_sample=200
    ######################################
    #####################################################Conditional DDIM generation
    model = DiffusionModel(net=net,n_steps=n_steps,max_epochs=max_epochs,lr=lr,scheduler=scheduler,ckpt_path=ckpt_path,line_lr_decay=line_lr_decay,continue_line_lr_decay=0.1,
                        continue_lr=4e-4,output_path=default_root_dir,cfg_scale=cfg_scale,save_interval=save_interval,show_training_max_samples=5,
                        ddim_steps=ddim_steps,simple_var=False, eta=0,no_cond_batch_n_sample=100,composition_tensor=composition_tensor,
                        per_composition_n_sample=per_composition_n_sample,show_samples_concat_and_animation=False,
                        autoencoder_path=autoencoder_path,model_id=model_id,latent_scale_factor=latent_scale_factor,
                        latent_shape=latent_shape,
                        beta_schedule = beta_schedule,
                        objective = objective,
                        min_snr_loss_weight = min_snr_loss_weight,
                        min_snr_gamma = min_snr_gamma,
                        p_drop=p_drop)
    trainer = pl.Trainer(max_epochs=max_epochs, devices=1, accelerator="gpu",strategy='ddp_find_unused_parameters_true',default_root_dir=default_root_dir)
    trainer.predict(model)
    
        


