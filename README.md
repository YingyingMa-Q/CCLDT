# CCLDT
Microstructure Generation of Ni-based Single-crystal Superalloys Across Continuous Composition Spaces via a Latent Diffusion Transformer
## Background
Establishing the generative composition–microstructure mapping is crucial for the design of nickel-based single-crystal superalloys. Here, based on the phase-field simulation data of the Ni–Al–Mo system, we implement a Composition-Conditioned Latent Diffusion Transformer (CCLDT) for the generalized generation of microstructure images within the continuous composition space. Built upon the Latent Diffusion Model (LDM) and Diffusion Transformer (DiT) architectures, the CCLDT incorporates an optimized compositional conditioning mechanism. It yields mean Intra-FID scores of 11.31 and 18.70 under seen and unseen compositions, respectively, demonstrating superior visual fidelity in the learned generative mapping.
## Ruirements
The versions of the pyhton library used are as follows:  
torch==2.5.1+cu121  
torch-fidelity==0.3.0  
torchmetrics==1.8.2  
torchvision==0.20.1+cu121  
tqdm==4.67.1  
opencv-python==4.12.0.88  
matplotlib==3.10.7  
numpy==2.1.2  
pandas==2.3.3  
pillow==11.3.0  
pytorch-fid==0.3.0  
pytorch-lightning==2.5.6  
scikit-learn==1.7.2  
## Code usage          
To train the model, first run preprocess_image.py for dataset splitting, then execute autoencoder_main.py to train the ACC-AE and save its parameters. Finally, run CCLDT_main.py to complete the DiT training and sampling process.
## Reference
The following repositories were used as inspiration for this implementation:  
DiT: https://github.com/facebookresearch/DiT/blob/main/models.py  
LDM: https://github.com/CompVis/latent-diffusion  
CCDM: https://github.com/UBCDingXin/CCDM/blob/main/CCDM_archived/CCDM_unified/diffusion.py  
## Citation
Please cite our paper as follows:   
Ma Y, Zhao W, Li M, Du S, Ru Y, Li S, Gong S, Xu H. Microstructure generation of Ni-based single-crystal superalloys across continuous composition spaces via a latent diffusion transformer. Journal of Materials Research and Technology 2026;42:12590–608. https://doi.org/10.1016/j.jmrt.2026.06.042
