import torch
torch.set_float32_matmul_precision('high')
import os
os.environ['NCCL_TIMEOUT'] = '7200000'
os.environ['TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC'] = '7200'
###############
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn.functional as F

from autoencoder_model import Encoder, Decoder
from distributions import DiagonalGaussianDistribution

from preprocess_image import train_dataloader, val_dataloader
import torchvision.utils
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF  
import os
from ae_loss_function import SimpleVAELoss
from pytorch_lightning.callbacks import ModelCheckpoint


class DelayedBestModelCheckpoint(ModelCheckpoint):
    """Callback for delayed saving of the best model."""
    def __init__(self, start_epoch_ratio=0.4, **kwargs):
        super().__init__(**kwargs)
        self.start_epoch_ratio = start_epoch_ratio
        self.start_epoch = None
        self._enabled = False  

    def on_train_start(self, trainer, pl_module):
        """Calculate the actual start_epoch at the beginning of training."""
        super().on_train_start(trainer, pl_module)
        self.start_epoch = int(trainer.max_epochs * self.start_epoch_ratio)
        print(f"[DelayedBestModelCheckpoint] 延迟保存: "
              f"从第 {self.start_epoch} 个 epoch 开始监控最佳模型")

    def on_validation_epoch_start(self, trainer, pl_module):
        """Determine whether to enable saving at the start of validation."""
        super().on_validation_epoch_start(trainer, pl_module)
        current_epoch = trainer.current_epoch
        if self.start_epoch is None:
            self.start_epoch = int(trainer.max_epochs * self.start_epoch_ratio)
        self._enabled = (current_epoch >= self.start_epoch)

        if self._enabled:
            print(f"[DelayedBestModelCheckpoint] Epoch {current_epoch}: 启用最佳模型检查")
        else:
            print(f"[DelayedBestModelCheckpoint] Epoch {current_epoch}: 禁用最佳模型检查")

    def on_validation_end(self, trainer, pl_module):
        """ecide whether to execute the saving process at the end of validation."""
        if not self._enabled:
            print(f"[DelayedBestModelCheckpoint] Epoch {trainer.current_epoch}: 跳过保存")
            return
        super().on_validation_end(trainer, pl_module)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Override the checkpoint saving method."""
        if not self._enabled:
            print(f"[DelayedBestModelCheckpoint] Epoch {trainer.current_epoch}: 阻止保存检查点")
            return
        print(f"[DelayedBestModelCheckpoint] Epoch {trainer.current_epoch}: 允许保存检查点")
        super().on_save_checkpoint(trainer, pl_module, checkpoint)
    
    def _save_checkpoint(self, trainer, filepath):
        """Override the checkpoint saving method."""
        if not self._enabled:
            print(f"[DelayedBestModelCheckpoint] 阻止保存: {filepath}")
            return
        print(f"[DelayedBestModelCheckpoint] 允许保存: {filepath}")
        super()._save_checkpoint(trainer, filepath)

    


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 save_interval=100,
                 max_epochs=500,
                 show_training_max_samples=6,
                 learning_rate=1e-3,
                 version_id='AE_v3'
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        ######################Initialize loss functions.
        self.lossconfig=lossconfig
        init_lossconfig = self.lossconfig.copy()  
        init_lossconfig["autocorr_weight"] = 0 
        self.loss_fn = SimpleVAELoss(**init_lossconfig)
        #####################
        assert ddconfig["double_z"]
        # Process feature maps: The encoder outputs z_channels * 2 to generate the "moments" (mean and variance) of the latent distribution.
        # Doubling the channels accounts for these distribution parameters. Note that embed_dim represents the channel count for mean/variance, which may differ from the latent variable's z_channels.
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        # Projects sampled latent vectors to z_channels for decoding.
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        
        self.z_resolution=ddconfig['resolution']//(2**len(ddconfig['ch_mult'])-1)
        self.max_epochs=max_epochs
        self.save_interval=save_interval if save_interval is not None else self.max_epochs//5
        self.show_training_max_samples=show_training_max_samples
        self.learning_rate=learning_rate
        self.version_id=version_id
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.ignore_keys=ignore_keys
        ############ saving settings
        self.save_hyperparameters()  
        self.should_save_this_epoch=False

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
    
    
    def encode(self, x):
        h = self.encoder(x)
        # Obtain the posterior distribution via the first moment (mean) and second moment (variance).
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        # Reconstructed images.
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            # Stochastic sampling (for training).
            z = posterior.sample()
        else:
            # Use the mean of the posterior distribution (for testing/inference).
            z = posterior.mode()
        dec = self.decode(z)
        # Return reconstructed data and posterior distributions.
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        return x
    def get_autocorr_weight(self, epoch, total_epochs):
        """Progressive weight scheduling for autocorrelation loss."""
        if epoch+1 < 0.05 * total_epochs:
            return 0.0  
        
        elif epoch+1 < 0.55 * total_epochs:
            start_epoch = 0.05 * total_epochs
            progress = (epoch+1 - start_epoch) / (0.5 * total_epochs)
            return self.lossconfig["autocorr_weight"] * progress  
        else:
            return self.lossconfig["autocorr_weight"]  
   
    
    def init_loss_weight(self):
        autocorr_weight = self.get_autocorr_weight(self.current_epoch,total_epochs=self.max_epochs)
        if hasattr(self.loss_fn, 'autocorr_weight'):
            self.loss_fn.autocorr_weight = autocorr_weight
        else:
            self.print(f"警告: loss_fn 缺少 autocorr_weight 属性")
        
   
    def on_train_epoch_start(self):
        self.init_loss_weight()
        self.log(
            "loss_weight/autocorr",  
            self.loss_fn.autocorr_weight,  
            on_step=False,  
            on_epoch=True, 
            prog_bar=False,  
            rank_zero_only=True  
        )
        
        current_epoch = self.current_epoch
        max_epochs = self.trainer.max_epochs
        save_interval = self.save_interval
        
        self.should_save_this_epoch = (
            (current_epoch + 1 == max_epochs) or 
            ((current_epoch + 1) % save_interval == 0)
        )
        

    
    def training_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        log_dict_ae = self.loss_fn(inputs, reconstructions, posterior)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=False, on_epoch=True,sync_dist=True,reduce_fx="mean")
        total_loss = log_dict_ae["train_loss/total"]
        lr = self.opt_ae.param_groups[0]['lr']
        self.log('learning_rate', lr,on_epoch=True,on_step=False,batch_size=inputs.shape[0], rank_zero_only=True)  
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            if self.should_save_this_epoch and \
                batch_idx == len(self.trainer.train_dataloader) - 1 and \
                self.trainer.is_global_zero:
                self.show_training_max_samples = min(inputs.shape[0], self.show_training_max_samples)
                x_subset = inputs[:self.show_training_max_samples]
                reco_x_subset,posterior = self.forward(x_subset)
                z_subset=posterior.mode()
                print("z shape",z_subset.shape)
                self.last_batch = {
                    "epoch":self.current_epoch,
                    'x':x_subset.detach().cpu(),
                    'z':z_subset.detach().cpu(),
                    'rec_x':reco_x_subset.detach().cpu()
                }    
            else:
                if hasattr(self, 'last_batch'):
                    del self.last_batch  
                    self.last_batch = None
        self.encoder.train()
        self.decoder.train()
        return total_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.encoder.eval()
        self.decoder.eval()
        inputs = self.get_input(batch, self.image_key)
        with torch.inference_mode():  
            reconstructions, posterior = self(inputs)
        valid_log_dict_ae = self.loss_fn(inputs, reconstructions, posterior,split='valid')
        # saved to TensorBoard 
        self.log_dict(valid_log_dict_ae, prog_bar=False, logger=True, on_step=False, on_epoch=True,sync_dist=True) 
        if self.should_save_this_epoch and batch_idx == len(self.trainer.val_dataloaders) - 1 and self.trainer.is_global_zero:
            self.show_training_max_samples = min(inputs.shape[0], self.show_training_max_samples)
            
            x_subset = inputs[:self.show_training_max_samples]
            reco_x_subset,posterior = self.forward(x_subset)
            z_subset=posterior.mode()
            self.last_valid_batch = {
                "epoch":self.current_epoch,
                'x':x_subset.detach().cpu(),
                'z':z_subset.detach().cpu(),
                'rec_x':reco_x_subset.detach().cpu()
            }    
        else:
            # self.last_valid_batch = None
            if hasattr(self, 'last_valid_batch'):
                del self.last_valid_batch  
                self.last_valid_batch = None
        return valid_log_dict_ae['valid_loss/total']


    def configure_optimizers(self):
        lr = self.learning_rate
        self.opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, 
                                  betas=(0.9, 0.999),
                                  weight_decay=0)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.opt_ae,
                start_factor=1,                     
                end_factor=0.1,                 
                total_iters=self.trainer.estimated_stepping_batches,            
            )
        return {
            "optimizer": self.opt_ae,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",  
            }
        }

    def get_last_layer(self):
        return self.decoder.conv_out.weight
    ########################################
    
    
    @torch.no_grad()
    def log_images_to_logger(self,last_batch):
        if not hasattr(self, 'last_batch') or not last_batch:
            return
        
        comparison_img, latent_img = self.create_visualizations(last_batch)
        
        
        for logger in self.trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                logger.experiment.add_image(
                    "Reconstruction_Comparison",
                    TF.to_tensor(comparison_img),
                    self.current_epoch
                )
                logger.experiment.add_image(
                    "Latent_Representations",
                    TF.to_tensor(latent_img),
                    self.current_epoch
                )
            
        

    #################Save images at specified epoch intervals.
    def create_visualizations(self, batch):
        """Create separate visualization grids for autoencoder results."""
        x = batch["x"]       # (6,1,128,128)
        x=(x+1.0)/2
        rec_x = batch["rec_x"] # (6,1,128,128)
        rec_x=(rec_x+1)/2
        z = batch["z"]         # (6,4,32,32)
        
        # Comparison between original and reconstructed images (2x6 grid).
        # --------------------------------------------------------
        # Concatenate original and reconstructed images (Shape: 12, 1, 128, 128).
        combined = torch.cat([x, rec_x], dim=0)
        
        comparison_grid = torchvision.utils.make_grid(
            combined,
            nrow=self.show_training_max_samples,            
            padding=10,
            pad_value=0.5,     
            normalize=False      
        )
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
        comparison_img = TF.to_pil_image(comparison_grid)
        comparison_height = comparison_img.height
        comparison_width = comparison_img.width
        new_comparison_img = Image.new("L", (comparison_width+150+10, comparison_height+20), color=220)
        new_comparison_img.paste(comparison_img, (150, 10))

        draw = ImageDraw.Draw(new_comparison_img)
        text_color = 0
        rect_color = 255
        title_height = 30
        img_half = x.shape[2] // 2
        draw.rectangle([0, 10+img_half-title_height//2, 150, 10+img_half+title_height//2], fill=rect_color)
        draw.rectangle([0,10+10+img_half*3-title_height//2 , 150, 10+10+img_half*3+title_height//2], fill=rect_color)

        draw.text((75, 10+img_half), 
                "Original Images", 
                fill=text_color, 
                font=font, 
                anchor="mm")

        draw.text((75, 10+10+img_half*3), 
                "Reconstructions", 
                fill=text_color, 
                font=font, 
                anchor="mm")
        
        font = ImageFont.load_default()
        z_up_res=64
        upsampled_z = F.interpolate(z, size=z_up_res, mode="bilinear")
        
        z_min = upsampled_z.amin(dim=(2,3), keepdim=True)
        z_max = upsampled_z.amax(dim=(2,3), keepdim=True)
        z_normalized = (upsampled_z - z_min) / (z_max - z_min + 1e-8)
        latent_flat = z_normalized.reshape(-1, 1, z_up_res, z_up_res)
        
        latent_grid = torchvision.utils.make_grid(
            latent_flat,
            nrow=self.show_training_max_samples,              
            padding=5,
            pad_value=0.2,        
            normalize=False,       
            scale_each=False        
        )
        
        latent_img = TF.to_pil_image(latent_grid)
        
        border_top = 40  
        border_left = 60  
        new_width = latent_img.width + border_left+10
        new_height = latent_img.height + border_top+10
        new_latent_img = Image.new("L", (new_width, new_height), color=220)  
        new_latent_img.paste(latent_img, (border_left, border_top))
        draw_latent = ImageDraw.Draw(new_latent_img)
        sample_cell_width = latent_img.width // self.show_training_max_samples
        for i in range(self.show_training_max_samples):
            x_pos = border_left + sample_cell_width * (i + 0.5)
            draw_latent.rectangle(
                [x_pos - 30, 5, x_pos + 30, border_top - 10],
                fill=rect_color,
                outline=text_color,
                width=1
            )
            draw_latent.text(
                (x_pos, border_top // 2), 
                f"Sample {i+1}", 
                fill=text_color, 
                font=font, 
                anchor="mm"  
            )
        
        ch_cell_height = latent_img.height // z.shape[1]
        for ch in range(z.shape[1]):
            y_pos = border_top + ch_cell_height * (ch + 0.5)
            
            draw_latent.rectangle(
                [10, y_pos - 15, border_left - 10, y_pos + 15],
                fill=rect_color,
                outline=text_color,
                width=1
                )
            
            draw_latent.text(
                (border_left // 2, y_pos), 
                f"Chan {ch+1}", 
                fill=text_color, 
                font=font, 
                anchor="mm"  
                )
        
        
        return new_comparison_img, new_latent_img


    ######################
    def on_train_epoch_end(self):
        if (self.should_save_this_epoch) and self.trainer.is_global_zero:
            self.log_images_to_logger(self.last_batch)
            if not hasattr(self, 'last_batch') or not self.last_batch:
                return
            comparison_img, latent_img = self.create_visualizations(self.last_batch)
            del self.last_batch
            save_dir = os.path.join(self.trainer.default_root_dir, f"saved-images/AE/{self.version_id}")
            os.makedirs(save_dir, exist_ok=True)
            
            comp_path = os.path.join(save_dir, f"train_recon_epoch{self.current_epoch}.png")
            latent_path = os.path.join(save_dir, f"train_latent_epoch{self.current_epoch}.png")
            
            comparison_img.save(comp_path)
            latent_img.save(latent_path)
            del comparison_img, latent_img  

        if hasattr(self, 'last_batch'):
            del self.last_batch
            

    def on_validation_epoch_end(self):
        if (self.should_save_this_epoch) and self.trainer.is_global_zero:
            self.log_images_to_logger(self.last_valid_batch)
            if not hasattr(self, 'last_valid_batch') or not self.last_valid_batch:
                return
            comparison_img, latent_img = self.create_visualizations(self.last_valid_batch)
            save_dir = os.path.join(self.trainer.default_root_dir, f"saved-images/AE/{self.version_id}")
            os.makedirs(save_dir, exist_ok=True)
            
            comp_path = os.path.join(save_dir, f"valid_recon_epoch{self.current_epoch}.png")
            latent_path = os.path.join(save_dir, f"valid_latent_epoch{self.current_epoch}.png")
            
            comparison_img.save(comp_path)
            latent_img.save(latent_path)
            del comparison_img, latent_img
        if hasattr(self, 'last_valid_batch'):
            del self.last_valid_batch




if __name__=='__main__':
    ckpt_path=None
    max_epochs=300
    save_interval=50
    show_training_max_samples=6
    default_root_dir="PhaseSimuDiffusionModel/time60_cond_ddpm/DITs"
    lossconfig={"kl_weight":1e-6, "pixel_loss_weight":1, "perceptual_weight":1.0,"autocorr_weight":3,
                 "threshold_type":'otsu',"max_lag":64}
    ddconfig={'double_z': True,
      'z_channels': 4,
      # original resolution
      'resolution': 256,
      'in_channels': 1,
      'out_ch': 1,
      'ch': 64,
      'ch_mult': [ 1,2,4,8 ],  # num_down = len(ch_mult)-1
      'num_res_blocks': 2,
      'attn_resolutions': [ ],
      'dropout': 0.0}
    embed_dim=4
    learning_rate=1e-4
    version_id='AE_v9_2'
    autoencoder = AutoencoderKL(ddconfig=ddconfig,
                 lossconfig=lossconfig,
                 embed_dim=embed_dim,
                 ckpt_path=ckpt_path,
                 ignore_keys=['loss_fn'],
                 image_key="image",
                 save_interval=save_interval,
                 max_epochs=max_epochs,
                 show_training_max_samples=show_training_max_samples,
                 learning_rate=learning_rate,
                 version_id=version_id
                 )
    ######
    save_dir = os.path.join(default_root_dir, f"saved-models/AE/{version_id}")
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_callback = DelayedBestModelCheckpoint(
            start_epoch_ratio=0.5,
            monitor="valid_loss/total",
            mode="min",
            save_top_k=1,
            filename="best-{epoch:02d}",
            dirpath=save_dir,  
            save_last=False,
            every_n_epochs=1,
            verbose=True
        )
    
    last_model_callback = ModelCheckpoint(
        monitor=None,
        save_top_k=1,  
        save_last=True,  
        filename="last-{epoch}",
        dirpath=save_dir
    )
    logger = TensorBoardLogger(
    save_dir="/home/ying/code/PhaseSimuDiffusionModel/time60_cond_ddpm/DITs/lightning_logs",  
    name="AE",  
    # version=10  
    )
    trainer = pl.Trainer(max_epochs=max_epochs, devices=2, accelerator="gpu",strategy='ddp_find_unused_parameters_false',default_root_dir=default_root_dir,
                            gradient_clip_val=1.0,
                            gradient_clip_algorithm="norm",
                            callbacks=[checkpoint_callback, last_model_callback],
                            logger=logger,
                        )
    trainer.fit(autoencoder, train_dataloader,val_dataloaders=val_dataloader)
    
    