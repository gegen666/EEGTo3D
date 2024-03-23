# 3D-Telepathy: EEG-to-3D using 2D Diffusion
## 1.Abstract
Reconstructing 3D visual stimulus scenes from electroencephalogram (EEG) data holds significant potential applications in brain-computer interface and aiding expression in populations with communication impairments. As of now, no work on EEG to 3D conversion has been proposed. The principal challenge is EEG itself has numerous sources of noise, rendering it challenging to extract visual 3D information from it. To address the groundbreaking task of EEG-to-3D reconstruction, we use dual masked contrastive learning and multimodal joint self-supervised learning to obtain the EEG encoder. By leveraging 2D diffusion as a prior distribution and training neural radiance fields (NeRF) through variational score distillation (VSD), we obtain the 3D scenes with both visually pleasing effects and well-defined structures from EEG. Additionally, during the 3D generation, we could obtain a set of images depicting the same object from varied  perspectives. This implies that we can derive multi-view visual and 3D-space information from EEG data that originally contained only single-view visual information. Overall, our work inaugurates a novel investigative trajectory by integrating the fields of neural signal decoding and 3D generation. It also offers novel insights into the neural representations of human perception of three-dimensional space.
## 2.Method Overview
![eegEncoder4](https://github.com/gegen666/EEGTo3D/assets/113605829/78255d41-52d1-42d7-9c30-6e0b3095fc67)

![model4](https://github.com/gegen666/EEGTo3D/assets/113605829/1ab7edb0-03c9-4e6a-a579-6a189de55344)

## 3.Training Procedure
### 3.1 MERL Stage1
### 3.2 MERL Stage2
### 3.3 Refine U-Net
### 3.4 3D Generation
