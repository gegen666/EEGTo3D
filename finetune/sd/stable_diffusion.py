
from finetune.utils.Condition_dataloader import image_dataloader
import argparse
from diffusers import (DDPMScheduler, UNet2DConditionModel,
                       AutoencoderKL, ControlNetModel,
                       StableDiffusionControlNetPipeline)
from finetune.diy_pipeline.C_Pipeline import C_Pipeline
import torch
from finetune.utils.common import load_config, get_parameters, copy_yaml_to_folder, clip_encode_images
from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image
from os.path import join as j
from transformers import CLIPTextModel, CLIPTokenizer
from config import Config_Generative_Model
from diffusers import StableDiffusionPipeline


def main(args):
    cf = load_config(args.config_path)
    copy_yaml_to_folder(args.config_path, cf['project_dir'])
    train_dataloader, val_dataloader = image_dataloader(cf['data_path'],
                                        cf['train_bc'], 
                                        cf['eval_bc'], 
                                        True)
    accelerator = Accelerator(**get_parameters(Accelerator, cf))
    autoencoderkl_tar = AutoencoderKL.from_pretrained(cf['target_path'])
    autoencoderkl_tar = autoencoderkl_tar.eval().to(accelerator.device)
    
    tokenizer = CLIPTokenizer.from_pretrained(cf['stable_path'], subfolder='tokenizer')
    text_encoder = CLIPTextModel.from_pretrained(cf['stable_path'], subfolder="text_encoder", use_safetensors=False)
    text_encoder = text_encoder.to(accelerator.device)
    text_encoder.requires_grad_(False)
    autoencoderkl_tar.requires_grad_(False)
    prompt_word = ''
    prompt = [prompt_word] * cf['train_bc']
    text_input = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )

    text_embeddings = text_encoder(text_input.input_ids.to(accelerator.device))[0]
        
    model = UNet2DConditionModel.from_pretrained(cf['unet_resume'])
    noise_scheduler = DDPMScheduler(**get_parameters(DDPMScheduler, cf))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cf["lr"])
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(train_dataloader) * cf['num_epochs']),
    )
    val_interval = cf['val_inter']
    save_interval = cf['save_inter']
    
    if len(cf['log_with']):
        accelerator.init_trackers('train_example')
    model, optimizer, train_dataloader, lr_scheduler, val_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler, val_dataloader
        )
    
    global_step = 0
    scaling_factor = autoencoderkl_tar.config.scaling_factor
    for epoch in range(cf['num_epochs']):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        
        for step, batch in enumerate(train_dataloader):
            targets  = batch[0]
            targets = targets.expand(cf['train_bc'], -1, -1, -1)
            print(f'target {targets.shape}')
            with accelerator.accumulate(model):
                latents = autoencoderkl_tar.encode(targets).latent_dist.sample()
                # print('latent shape: ', latents.shape)
                latents = latents * scaling_factor

                bs = latents.shape[0]
                noise = torch.randn_like(latents, device=latents.device)
                
                timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                model_pred = model(noisy_latents, timesteps, text_embeddings).sample
                loss = F.mse_loss(model_pred.float(), noise.float(), reduce='mean')
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs, refresh=False)
            progress_bar.update(1)
            accelerator.log(logs, step=global_step)
            global_step += 1
            
        if accelerator.is_main_process:
            pipeline = StableDiffusionPipeline(autoencoderkl_tar, 
                                               text_encoder, 
                                               tokenizer, 
                                               accelerator.unwrap_model(model), 
                                               noise_scheduler, 
                                               None, 
                                               None, 
                                               False)
            if (epoch + 1) % val_interval == 0 or epoch == cf['num_epochs'] - 1:
                val_data = next(iter(val_dataloader))
                targets  = val_data[0]
                targets = targets.expand(cf['train_bc'], -1, -1, -1)
                print(f'target {targets.shape}')
                images = pipeline([prompt_word]*cf['eval_bc'], 
                                  num_inference_steps=100, 
                                  guidance_scale=1.0,
                                  output_type='pt').images
                images = torch.cat([(images-0.5)/0.5, targets], dim=-1)
                images = (make_grid(images, nrow=1).unsqueeze(0) + 1) / 2 
                accelerator.trackers[0].log_images({'Generate': images.cpu().detach()}, epoch+1) 
                
            if (epoch + 1) % save_interval == 0 or epoch == cf['num_epochs'] - 1:
                pipeline.save_pretrained(cf['project_dir'])  
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='/config/stable_diffusion_config.yaml')

    args = parser.parse_args()
    main(args)