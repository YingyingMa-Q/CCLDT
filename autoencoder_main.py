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

# from util import instantiate_from_config
from preprocess_image import train_dataloader, val_dataloader
import torchvision.utils
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF  # 用于张量和PIL图像之间的转换
import os
# from contperceptual import SimpleVAELoss
from ae_loss_function import SimpleVAELoss
from pytorch_lightning.callbacks import ModelCheckpoint


class DelayedBestModelCheckpoint(ModelCheckpoint):
    """延迟保存最佳模型的回调"""
    def __init__(self, start_epoch_ratio=0.4, **kwargs):
        super().__init__(**kwargs)
        self.start_epoch_ratio = start_epoch_ratio
        self.start_epoch = None
        self._enabled = False  # 是否启用最佳模型保存

    def on_train_start(self, trainer, pl_module):
        """训练开始时计算真正的 start_epoch"""
        super().on_train_start(trainer, pl_module)
        self.start_epoch = int(trainer.max_epochs * self.start_epoch_ratio)
        print(f"[DelayedBestModelCheckpoint] 延迟保存: "
              f"从第 {self.start_epoch} 个 epoch 开始监控最佳模型")

    def on_validation_epoch_start(self, trainer, pl_module):
        """验证开始时决定是否启用保存"""
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
        """验证结束时决定是否执行保存"""
        if not self._enabled:
            print(f"[DelayedBestModelCheckpoint] Epoch {trainer.current_epoch}: 跳过保存")
            return
        super().on_validation_end(trainer, pl_module)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """覆盖保存检查点方法 - 关键修复点"""
        if not self._enabled:
            print(f"[DelayedBestModelCheckpoint] Epoch {trainer.current_epoch}: 阻止保存检查点")
            return
        print(f"[DelayedBestModelCheckpoint] Epoch {trainer.current_epoch}: 允许保存检查点")
        super().on_save_checkpoint(trainer, pl_module, checkpoint)
    
    def _save_checkpoint(self, trainer, filepath):
        """覆盖内部保存方法 - 最关键的修复点"""
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
        ######################初始化损失函数
        self.lossconfig=lossconfig
        init_lossconfig = self.lossconfig.copy()  # 创建配置副本
        init_lossconfig["autocorr_weight"] = 0  # 更新为当前值
        self.loss_fn = SimpleVAELoss(**init_lossconfig)
        # 延迟创建损失函数
        # self.loss_fn = None  
        #####################
        assert ddconfig["double_z"]
        # 处理特征图，编码器的输出是z_channels*2，生成表示潜在分布参数的"矩"，即均值和方差，因此是双倍通道，embed_dim表示均值和方差的通道数，可以和潜在变量的通道数不一样
        # 例如将embed_dim改成1或者3可以进行可视化
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        # 将从后验分布中抽取的样本转换为z_channels，为输入解码器做准备
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        #####新加的参数
        # 潜在变量的分辨率
        self.z_resolution=ddconfig['resolution']//(2**len(ddconfig['ch_mult'])-1)
        self.max_epochs=max_epochs
        self.save_interval=save_interval if save_interval is not None else self.max_epochs//5
        self.show_training_max_samples=show_training_max_samples
        self.learning_rate=learning_rate
        self.version_id=version_id
        ###########NOTE 这种是加载权重，但优化器，scheduler 状态，epoch都不会恢复
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.ignore_keys=ignore_keys
        ############ TODO 保存配置
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
        # 即一阶矩均值、二阶矩方差，得到后验分布
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        # 重建的图像
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            # 随机采样（用于训练）
            z = posterior.sample()
        else:
            # 取后验分布的均值（用于测试/推理）
            z = posterior.mode()
        dec = self.decode(z)
        # 返回重建数据和后验分布
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        return x
    def get_autocorr_weight(self, epoch, total_epochs):
        """自相关损失的渐进式权重调度"""
        # 阶段1：基础重建（前20%训练周期）
        if epoch+1 < 0.05 * total_epochs:
            return 0.0  # 完全禁用
        
        # TODO 阶段2：由0.2~0.8训练周期渐进引入，改成v92 0.2~0.4渐进引入，v93改成0.1~0.2，v5改成0.1~0.6
        elif epoch+1 < 0.55 * total_epochs:
            start_epoch = 0.05 * total_epochs
            progress = (epoch+1 - start_epoch) / (0.5 * total_epochs)
            return self.lossconfig["autocorr_weight"] * progress  # 线性增长至1
        # 阶段3：稳定强化（后60%训练周期）
        else:
            return self.lossconfig["autocorr_weight"]   # 目标权重
   
    
    def init_loss_weight(self):
        autocorr_weight = self.get_autocorr_weight(self.current_epoch,total_epochs=self.max_epochs)
        # kl_weight = self.get_kl_weight(self.current_epoch,total_epochs=self.max_epochs)
        # 直接更新现有loss_fn的权重
        if hasattr(self.loss_fn, 'autocorr_weight'):
            self.loss_fn.autocorr_weight = autocorr_weight
        else:
            self.print(f"警告: loss_fn 缺少 autocorr_weight 属性")
        
    # 在on_train_epoch_start中更新一次
    def on_train_epoch_start(self):
        self.init_loss_weight()
        self.log(
            "loss_weight/autocorr",  # 日志中的键名（使用/分组）
            self.loss_fn.autocorr_weight,  # 要记录的权重值
            on_step=False,  # 不在每个训练step记录
            on_epoch=True,  # 在每个epoch记录
            prog_bar=False,  # 不在进度条显示（可选）
            rank_zero_only=True  # 关键参数：只在rank 0记录
        )
        
        # 计算保存条件
        current_epoch = self.current_epoch
        max_epochs = self.trainer.max_epochs
        save_interval = self.save_interval
        
        # 判断是否需要保存
        self.should_save_this_epoch = (
            (current_epoch + 1 == max_epochs) or 
            ((current_epoch + 1) % save_interval == 0)
        )
        

    
    def training_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        # self(inputs)是调用前向传播
        reconstructions, posterior = self(inputs)
        log_dict_ae = self.loss_fn(inputs, reconstructions, posterior)
        # 将字典中的所有指标记录到 TensorBoard,v60之前记录的损失是step损失，改成epoch损失
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=False, on_epoch=True,sync_dist=True,reduce_fx="mean")# 确保epoch值是所有step的平均
        # 获取总损失用于返回
        total_loss = log_dict_ae["train_loss/total"]
        lr = self.opt_ae.param_groups[0]['lr']
        self.log('learning_rate', lr,on_epoch=True,on_step=False,batch_size=inputs.shape[0], rank_zero_only=True)  # 记录到TensorBoard
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            ############TODO 只在满足需要保存的epoch才保存数据
            # 只有在满足保存条件且是最后一个批次且是主进程时才保存数据
            if self.should_save_this_epoch and \
                batch_idx == len(self.trainer.train_dataloader) - 1 and \
                self.trainer.is_global_zero:
                # 确定需要保存的最大样本数
                self.show_training_max_samples = min(inputs.shape[0], self.show_training_max_samples)
                
                # --- 仅处理需要保存的样本 ---
                # 切片获取前 max_samples 个样本
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
                    del self.last_batch  # 关键步骤
                    self.last_batch = None
        self.encoder.train()
        self.decoder.train()
        return total_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # 确保模型处于评估状态（重要！）
        self.encoder.eval()
        self.decoder.eval()
        inputs = self.get_input(batch, self.image_key)
        # 前向传播（需要添加评估模式）
        with torch.inference_mode():  # 比 torch.no_grad() 更快
            reconstructions, posterior = self(inputs)
        valid_log_dict_ae = self.loss_fn(inputs, reconstructions, posterior,split='valid')
        # 将字典中的所有指标记录到 TensorBoard 
        self.log_dict(valid_log_dict_ae, prog_bar=False, logger=True, on_step=False, on_epoch=True,sync_dist=True) # 多GPU同步
        if self.should_save_this_epoch and batch_idx == len(self.trainer.val_dataloaders) - 1 and self.trainer.is_global_zero:
            # 确定需要保存的最大样本数
            self.show_training_max_samples = min(inputs.shape[0], self.show_training_max_samples)
            
            # --- 仅处理需要保存的样本 ---
            # 切片获取前 max_samples 个样本
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
                del self.last_valid_batch  # 关键步骤
                self.last_valid_batch = None
        return valid_log_dict_ae['valid_loss/total']


    def configure_optimizers(self):
        lr = self.learning_rate
        self.opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, 
                                   # 默认值是0.9，0.999，较低的 β₁ (0.5)​​：​直接影响​​：更少依赖历史梯度，更大权重给当前梯度，
                                   # 较高的beta2保持较短的历史梯度幅值记忆,自适应学习率对近期梯度变化更敏感
                                #   betas=(0.5, 0.9),
                                  betas=(0.9, 0.999),
                                  weight_decay=0)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.opt_ae,
                start_factor=1,                      # 初始学习率因子
                ###NOTE vAE 59由0.01调整为0.1
                end_factor=0.1,                 
                total_iters=self.trainer.estimated_stepping_batches,            
            )
        # return [self.opt_ae], [self.scheduler]
        # 返回包含梯度裁剪的字典
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
            
        

    #################间隔指定的epochs保存图像
    def create_visualizations(self, batch):
        """为自编码器结果创建分离的可视化网格"""
        # 只使用前6个样本（保证布局整洁）
        x = batch["x"]       # (6,1,128,128)
        x=(x+1.0)/2
        rec_x = batch["rec_x"] # (6,1,128,128)
        rec_x=(rec_x+1)/2
        z = batch["z"]         # (6,4,32,32)
        
        # 1. 原始图像与重建图像对比 (2×6网格)
        # --------------------------------------------------------
        # 合并原始和重建图像 (12,1,128,128)
        combined = torch.cat([x, rec_x], dim=0)
        
        # 创建2×6网格 (原始在第一行，重建在第二行)
        comparison_grid = torchvision.utils.make_grid(
            combined,
            nrow=self.show_training_max_samples,            # 每行6个样本
            padding=10,
            pad_value=0.5,     # 灰色分隔线
            normalize=False      # 自动归一化到[0,1]
        )
        
        # 添加标题文本（使用PIL）
         # 加载字体
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            # print('---------DejaVuSans-Bold字体----------')
        except:
            font = ImageFont.load_default()
        comparison_img = TF.to_pil_image(comparison_grid)
        # 为对比图添加顶部边距放置标题
        comparison_height = comparison_img.height
        comparison_width = comparison_img.width
        new_comparison_img = Image.new("L", (comparison_width+150+10, comparison_height+20), color=220)
        # 粘贴原始对比网格
        new_comparison_img.paste(comparison_img, (150, 10))

        draw = ImageDraw.Draw(new_comparison_img)
        # 灰度图中只能使用灰度值
        text_color = 0
        rect_color = 255
        # 在顶部边距添加标题区域
        title_height = 30
        img_half = x.shape[2] // 2
        # 左上角x,y，右下角x,y
        draw.rectangle([0, 10+img_half-title_height//2, 150, 10+img_half+title_height//2], fill=rect_color)
        draw.rectangle([0,10+10+img_half*3-title_height//2 , 150, 10+10+img_half*3+title_height//2], fill=rect_color)
        # 添加标题文本
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
        
        # 独立归一化每个通道到[0,1]
        z_min = upsampled_z.amin(dim=(2,3), keepdim=True)
        z_max = upsampled_z.amax(dim=(2,3), keepdim=True)
        z_normalized = (upsampled_z - z_min) / (z_max - z_min + 1e-8)
        latent_flat = z_normalized.reshape(-1, 1, z_up_res, z_up_res)
        
        # 创建4×6网格 (4行通道，6列样本)
        latent_grid = torchvision.utils.make_grid(
            latent_flat,
            nrow=self.show_training_max_samples,               # 每行6个样本
            padding=5,
            pad_value=0.2,        # 浅灰色分隔
            normalize=False,       
            scale_each=False        
        )
        
        # 添加标题文本
        latent_img = TF.to_pil_image(latent_grid)
        
        # 创建带有扩展画布的新图像 (为文本添加边距)
        border_top = 40  # 顶部边距 (样本标题)
        border_left = 60  # 左侧边距 (通道标题)
        new_width = latent_img.width + border_left+10
        new_height = latent_img.height + border_top+10
        new_latent_img = Image.new("L", (new_width, new_height), color=220)  # 浅灰背景
        # 粘贴潜在空间网格
        new_latent_img.paste(latent_img, (border_left, border_top))
        # 创建Draw对象
        draw_latent = ImageDraw.Draw(new_latent_img)
        # 添加列标题（样本编号） - 放在每个样本上方
        sample_cell_width = latent_img.width // self.show_training_max_samples
        for i in range(self.show_training_max_samples):
            x_pos = border_left + sample_cell_width * (i + 0.5)
            # 添加背景框提高可读性
            draw_latent.rectangle(
                [x_pos - 30, 5, x_pos + 30, border_top - 10],
                fill=rect_color,
                outline=text_color,
                width=1
            )
            # 在顶部边距区添加样本标题
            draw_latent.text(
                (x_pos, border_top // 2), 
                f"Sample {i+1}", 
                fill=text_color, 
                font=font, 
                anchor="mm"  # 中心对齐
            )
        
        # 添加行标题（通道编号） - 放在每行左侧
        ch_cell_height = latent_img.height // z.shape[1]
        for ch in range(z.shape[1]):
            # 计算y位置：原始网格中通道位置 + 顶部边距偏移
            y_pos = border_top + ch_cell_height * (ch + 0.5)
            
            # 添加背景框提高可读性
            draw_latent.rectangle(
                [10, y_pos - 15, border_left - 10, y_pos + 15],
                fill=rect_color,
                outline=text_color,
                width=1
                )
            
            # 在左侧边距区添加通道标题
            draw_latent.text(
                (border_left // 2, y_pos), 
                f"Chan {ch+1}", 
                fill=text_color, 
                font=font, 
                anchor="mm"  # 中心对齐
                )
        
        
        return new_comparison_img, new_latent_img


    ######################
    def on_train_epoch_end(self):
        # 判断是否满足保存条件
        # is_last_epoch = (self.current_epoch + 1) == self.trainer.max_epochs
        # is_interval_epoch = ((self.current_epoch+1) % self.save_interval) == 0
        if (self.should_save_this_epoch) and self.trainer.is_global_zero:
            self.log_images_to_logger(self.last_batch)
            if not hasattr(self, 'last_batch') or not self.last_batch:
                return
            comparison_img, latent_img = self.create_visualizations(self.last_batch)
            del self.last_batch
            # 保存图像到磁盘（解决无法点击保存的问题）
            save_dir = os.path.join(self.trainer.default_root_dir, f"saved-images/AE/{self.version_id}")
            os.makedirs(save_dir, exist_ok=True)
            
            comp_path = os.path.join(save_dir, f"train_recon_epoch{self.current_epoch}.png")
            latent_path = os.path.join(save_dir, f"train_latent_epoch{self.current_epoch}.png")
            
            comparison_img.save(comp_path)
            latent_img.save(latent_path)
            del comparison_img, latent_img  # 释放图像对象

        # 确保last_batch被删除
        if hasattr(self, 'last_batch'):
            del self.last_batch
            

    def on_validation_epoch_end(self):
        # 判断是否满足保存条件
        # is_last_epoch = (self.current_epoch + 1) == self.trainer.max_epochs
        # is_interval_epoch = ((self.current_epoch+1) % self.save_interval) == 0
        if (self.should_save_this_epoch) and self.trainer.is_global_zero:
            self.log_images_to_logger(self.last_valid_batch)
            if not hasattr(self, 'last_valid_batch') or not self.last_valid_batch:
                return
            comparison_img, latent_img = self.create_visualizations(self.last_valid_batch)
            # 保存图像到磁盘（解决无法点击保存的问题）
            save_dir = os.path.join(self.trainer.default_root_dir, f"saved-images/AE/{self.version_id}")
            os.makedirs(save_dir, exist_ok=True)
            
            comp_path = os.path.join(save_dir, f"valid_recon_epoch{self.current_epoch}.png")
            latent_path = os.path.join(save_dir, f"valid_latent_epoch{self.current_epoch}.png")
            
            comparison_img.save(comp_path)
            latent_img.save(latent_path)
            # 立即释放不再需要的对象
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
      # 填写原始图像高度（宽度），但不会进行数据检查
      'resolution': 256,
      # in_channels: 3
      'in_channels': 1,
      # out_ch: 3
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
            # 最小宽度为2，不足补0
            filename="best-{epoch:02d}",
            dirpath=save_dir,  # 保存目录
            save_last=False,
            every_n_epochs=1,
            verbose=True
        )
    # 添加另一个回调保存最后模型
    last_model_callback = ModelCheckpoint(
        monitor=None,
        save_top_k=1,  
        save_last=True,  # 保存最后一个模型
        filename="last-{epoch}",
        dirpath=save_dir
    )
    logger = TensorBoardLogger(
    save_dir="/home/ying/code/PhaseSimuDiffusionModel/time60_cond_ddpm/DITs/lightning_logs",  # 日志保存目录
    name="AE",  # 实验名称
    # version=10  # 可选：指定版本号
    )
    trainer = pl.Trainer(max_epochs=max_epochs, devices=2, accelerator="gpu",strategy='ddp_find_unused_parameters_false',default_root_dir=default_root_dir,
                            ###在这里添加梯度裁剪
                            gradient_clip_val=1.0,
                            gradient_clip_algorithm="norm",
                            callbacks=[checkpoint_callback, last_model_callback],
                            logger=logger,
                        )
    trainer.fit(autoencoder, train_dataloader,val_dataloaders=val_dataloader)
    
    