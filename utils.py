import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Noto Sans CJK JP', 'Microsoft YaHei']  
plt.rcParams['axes.unicode_minus'] = False  
from matplotlib.animation import FuncAnimation
import os


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t


def generate_animation(intermediate_samples, t_steps, fname, n_images_per_row=8):
    """Generates animation and saves as a gif file for given intermediate samples"""
    intermediate_samples = [make_grid(x, scale_each=True, normalize=True, 
                                      nrow=n_images_per_row).permute(1, 2, 0).numpy() for x in intermediate_samples]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis("off")
    img_plot = ax.imshow(intermediate_samples[0])
    
    def update(frame):
        img_plot.set_array(intermediate_samples[frame])
        ax.set_title(f"T = {t_steps[frame]}")
        fig.tight_layout()
        return img_plot
    
    ani = FuncAnimation(fig, update, frames=len(intermediate_samples), interval=200)
    ani.save(fname)




