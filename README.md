# 3D-Telepathy: EEG-to-3D using 2D Diffusion

![3D-Telepathy-Overview](https://github.com/gegen666/EEGTo3D/assets/113605829/c68a54ad-7974-47d8-90f6-7ed27acdbdb3)

## 1.Abstract
Reconstructing 3D visual stimulus scenes from electroencephalogram (EEG) data holds significant potential applications in brain-computer interface and aiding expression in populations with communication impairments. As of now, no work on EEG to 3D conversion has been proposed. The principal challenge is EEG itself has numerous sources of noise, rendering it challenging to extract visual 3D information from it. To address the groundbreaking task of EEG-to-3D reconstruction, we use dual masked contrastive learning and multimodal joint self-supervised learning to obtain the EEG encoder. By leveraging 2D diffusion as a prior distribution and training neural radiance fields (NeRF) through variational score distillation (VSD), we obtain the 3D scenes with both visually pleasing effects and well-defined structures from EEG. Additionally, during the 3D generation, we could obtain a set of images depicting the same object from varied  perspectives. This implies that we can derive multi-view visual and 3D-space information from EEG data that originally contained only single-view visual information. Overall, our work inaugurates a novel investigative trajectory by integrating the fields of neural signal decoding and 3D generation. It also offers novel insights into the neural representations of human perception of three-dimensional space.
## 2.Method Overview
![eegEncoder4](https://github.com/gegen666/EEGTo3D/assets/113605829/78255d41-52d1-42d7-9c30-6e0b3095fc67)

![model4](https://github.com/gegen666/EEGTo3D/assets/113605829/1ab7edb0-03c9-4e6a-a579-6a189de55344)

## 3.Preparatory work
The datasets folder and pretrains folder are not included in this repository. Please download the EEG data from [eeg](https://github.com/perceivelab/eeg_visual_classification). We also provide a copy of the Imagenet subset [imagenet](https://drive.google.com/file/d/1y7I9bG1zKYqBM94odcox_eQjnP9HGo9-/view?usp=drive_link).

Throughout the entire process, we utilized the standard SD1.5.You can sownload the parameter from [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5).

## 4.Training Procedure
### 4.1 MERL Stage1
```
python stageA1_mbm_pretrain_contrast.py \
--output_path . \
--lr 0.001 \ 
—-batch_size 10 \
—-do_self_contrast True \
—-do_cross_contrast True \
--self_contrast_loss_weight 1 \ 
--cross_contrast_loss_weight 0.5 \
—mask_ratio 0.75 \
—num_epoch 150 
```
### 4.2 MERL Stage2
```
python stageA2_mbm_finetune_cross.py \
--pretrain_mbm_path your_pretrained_ckpt_from_phase1 \
--batch_size 4 \
--num_epoch 60 \
--fmri_recon_weight 0.25 \ 
--img_recon_weight 1.5 \
--output_path your_output_path \ 
--img_mask_ratio 0.5 \
--mask_ratio 0.75 
```
### 4.3 Refine U-Net
```
python finetune/sd/train_sd_with_con.py
```
### 4.4 3D Generation
```
python 3dGeneration/main.py --iters 100000 --lambda_entropy 10 --scale 3 --n_particles 4 --h 512  --w 512 --t5_iters 20000 --workspace exp-nerf-stage1/
```
