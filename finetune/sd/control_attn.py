import sys; sys.path.append('./')
from utils.Condition_dataloader import double_form_dataloader
import argparse
from diffusers import (DDPMScheduler, UNet2DConditionModel, 
                       AutoencoderKL, 
                       ControlNetModel)
from diffusers import ControlNetModel
import torch
from utils.common import (load_config, get_parameters, 
                          copy_yaml_to_folder, force_remove_empty_dir)
from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image
from os.path import join as j
from os import path; import os
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.models.modeling_utils import ModelMixin
import torch
from diffusers.models.attention_processor import Attention, AttnProcessor, XFormersAttnProcessor, AttnProcessor2_0
from diffusers.utils.import_utils import is_xformers_available
from typing import Any
from diy_pipeline.C_Pipeline import Ctrl_Ref_Tensor

class ReferenceOnlyAttnProc(torch.nn.Module):
    def __init__(
        self,
        chained_proc,
        enabled=False,
        name=None
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.chained_proc = chained_proc
        self.name = name

    def __call__(
        self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None,
        mode="w", ref_dict: dict = None, is_cfg_guidance = False
    ) -> Any:
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        if self.enabled and is_cfg_guidance:
            res0 = self.chained_proc(attn, hidden_states[:1], encoder_hidden_states[:1], attention_mask)
            hidden_states = hidden_states[1:]
            encoder_hidden_states = encoder_hidden_states[1:]
        if self.enabled:
            if mode == 'w':
                ref_dict[self.name] = encoder_hidden_states
            elif mode == 'r':
                print(f"Shape of {self.name} is : {ref_dict[self.name].shape}")
                print(f"ehs shape: {encoder_hidden_states.shape}")
                # encoder_hidden_states = torch.cat([encoder_hidden_states, ref_dict.pop(self.name)], dim=1)
                encoder_hidden_states = ref_dict.pop(self.name).detach()
            elif mode == 'm':
                encoder_hidden_states = torch.cat([encoder_hidden_states, ref_dict[self.name]], dim=1)
            else:
                assert False, mode
        res = self.chained_proc(attn, hidden_states, encoder_hidden_states, attention_mask)
        if self.enabled and is_cfg_guidance:
            res = torch.cat([res0, res])
        return res


class Attn_from_crl(ModelMixin):
    def __init__(self, ref_unet: UNet2DConditionModel, train_unet:UNet2DConditionModel, ctrl_net:ControlNetModel):
        super().__init__()
        # make sure the structure of ref_unet and train_unet is the same !!!
        self.ref_unet = ref_unet
        self.train_unet = train_unet
        self.ctrl_net = ctrl_net
        
        unet_lora_attn_procs = dict()
        for name, _ in train_unet.attn_processors.items():
            if torch.__version__ >= '2.0':
                default_attn_proc = AttnProcessor2_0()
            elif is_xformers_available():
                default_attn_proc = XFormersAttnProcessor()
            else:
                default_attn_proc = AttnProcessor()
            unet_lora_attn_procs[name] = ReferenceOnlyAttnProc(
                default_attn_proc, enabled=name.endswith("attn1.processor"), name=name
            )
        self.train_unet.set_attn_processor(unet_lora_attn_procs)
        
        unet_lora_attn_procs = dict()
        for name, _ in ref_unet.attn_processors.items():
            if torch.__version__ >= '2.0':
                default_attn_proc = AttnProcessor2_0()
            elif is_xformers_available():
                default_attn_proc = XFormersAttnProcessor()
            else:
                default_attn_proc = AttnProcessor()
            unet_lora_attn_procs[name] = ReferenceOnlyAttnProc(
                default_attn_proc, enabled=name.endswith("attn1.processor"), name=name
            )
        self.ref_unet.set_attn_processor(unet_lora_attn_procs)
        
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.train_unet, name)
    
    @torch.no_grad()
    def ref_forward(self, noisy_sample, timestep, encoder_hidden_states, 
                    controlnet_cond, ref_dict, **kwargs):
        down_block_res_samples, mid_block_res_sample = self.ctrl_net(
                    noisy_sample, 
                    timestep, 
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_cond,
                    return_dict=False, 
                )
        
        pred = self.ref_unet(
            noisy_sample, timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            cross_attention_kwargs=dict(mode="w", ref_dict=ref_dict),
            **kwargs
        ).sample
        return pred # maybe need for evaluation
    
    def forward(
        self, sample_cond, sample_target, timestep, ehs_cond, ehs_target, cond, 
        down_block_res_samples=None, mid_block_res_sample=None, return_mid=False,
        **kwargs
    ):
        
        ref_dict = {}
        ref_pred = self.ref_forward(sample_cond, timestep, ehs_cond, 
                         cond, ref_dict, **kwargs)
        weight_dtype = self.train_unet.dtype
        train_pred = self.train_unet(
                sample_target, timestep, 
                ehs_target, 
                cross_attention_kwargs=dict(mode="r", ref_dict=ref_dict),
                down_block_additional_residuals=[
                    sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                ] if down_block_res_samples is not None else None,
                mid_block_additional_residual=(
                    mid_block_res_sample.to(dtype=weight_dtype)
                    if mid_block_res_sample is not None else None
                ),
                **kwargs
            ).sample
        if not return_mid:
            return train_pred
        else:
            return train_pred, ref_pred
        
        
def main(args):
    cf = load_config(args.config_path)
    if path.exists(cf['project_dir']): # previous failed traning
        force_remove_empty_dir(cf['project_dir'], )
    copy_yaml_to_folder(args.config_path, cf['project_dir'])
    train_dataloader = double_form_dataloader(cf['train_path'], 
                                        cf['img_sz'], 
                                        cf['train_bc'],
                                        cf['mode'])
    
    val_dataloader = double_form_dataloader(cf['eval_path'], 
                                        cf['img_sz'], 
                                        cf['eval_bc'],
                                        cf['mode'])
    
    accelerator = Accelerator(**get_parameters(Accelerator, cf))
    autoencoderkl_tar = AutoencoderKL.from_pretrained(cf['target_path'])
    autoencoderkl_tar = autoencoderkl_tar.eval().to(accelerator.device)
    
    
    tokenizer = CLIPTokenizer.from_pretrained(cf['stable_path'], subfolder='tokenizer')
    text_encoder = CLIPTextModel.from_pretrained(cf['stable_path'], subfolder="text_encoder", use_safetensors=False)
    text_encoder = text_encoder.to(accelerator.device)
    prompt_word = 'To generate an Ultra-Wide-angle Fundus Fluorescein Angiography from an Ultra-Wide-angle Fundus Scanning Laser Ophthalmoscopy'
    prompt = [prompt_word] * cf['train_bc']
    text_input = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )

    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(accelerator.device))[0]
    print("text_shape: ", text_embeddings.shape)
    
    ref_model = UNet2DConditionModel(sample_size=cf["sample_size"], 
                        in_channels=cf['in_channels'],
                        out_channels=cf['out_channels'], 
                        layers_per_block=cf['layers_per_block'],
                        block_out_channels=cf['block_out_channels'],
                        down_block_types=cf['down_block_types'], 
                        up_block_types=cf['up_block_types'], 
                        cross_attention_dim=cf['cross_attention_dim'])
    
    train_model = UNet2DConditionModel(sample_size=cf["sample_size"], 
                        in_channels=cf['in_channels'],
                        out_channels=cf['out_channels'], 
                        layers_per_block=cf['layers_per_block'],
                        block_out_channels=cf['block_out_channels'],
                        down_block_types=cf['down_block_types'], 
                        up_block_types=cf['up_block_types'], 
                        cross_attention_dim=cf['cross_attention_dim'])
    
    if len(cf['unet_resume']):
        ref_model = UNet2DConditionModel.from_pretrained(cf['unet_resume'])
        train_model = UNet2DConditionModel.from_pretrained(cf['unet_resume'])
        
        
    controlnet = ControlNetModel.from_pretrained(cf['ctrl_resume'])
    
    noise_scheduler = DDPMScheduler(**get_parameters(DDPMScheduler, cf))
    optimizer = torch.optim.AdamW(train_model.parameters(), lr=cf["lr"])
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(train_dataloader) * cf['num_epochs']),
    )
    ref_model.requires_grad_(False).eval()
    controlnet.requires_grad_(False).eval()
    autoencoderkl_tar.requires_grad_(False)
    text_encoder.requires_grad_(False)
    ref_model.to(accelerator.device)
    controlnet.to(accelerator.device)
    
    model = Attn_from_crl(ref_model, train_model, controlnet)
    
    val_interval = cf['val_inter']
    save_interval = cf['save_inter']
    if len(cf['log_with']):
        accelerator.init_trackers('train_example')
    train_model, optimizer, train_dataloader, lr_scheduler, val_dataloader = accelerator.prepare(
            train_model, optimizer, train_dataloader, lr_scheduler, val_dataloader
        )
    global_step = 0
    scaling_factor = autoencoderkl_tar.config.scaling_factor
    for epoch in range(cf['num_epochs']):
        train_model.train()
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        
        for step, batch in enumerate(train_dataloader):
            batch = batch[0]
            slo, first_ffa, targets = batch[0], batch[1], batch[2]
            with accelerator.accumulate(train_model):
                latents_tar = autoencoderkl_tar.encode(targets).latent_dist.sample()
                latents_tar = latents_tar * scaling_factor
                latents_cond = autoencoderkl_tar.encode(first_ffa).latent_dist.sample()
                latents_cond = latents_cond * scaling_factor
                bs = latents_tar.shape[0]
                noise = torch.randn_like(latents_tar)
                
                
                timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=latents_tar.device
                ).long()
                latents_tar = noise_scheduler.add_noise(latents_tar, noise, timesteps)
                latents_cond = noise_scheduler.add_noise(latents_cond, noise, timesteps)
                
                pred = model(latents_tar, latents_cond, timesteps, 
                             text_embeddings, text_embeddings, slo)
                loss = F.mse_loss(pred.float(), noise.float(), reduce='mean')
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(train_model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs, refresh=False)
            progress_bar.update(1)
            accelerator.log(logs, step=global_step)
            global_step += 1
    
        if accelerator.is_main_process:
            pipeline = Ctrl_Ref_Tensor(accelerator.unwrap_model(model), noise_scheduler)
            
            prompt = [prompt_word] * cf['eval_bc']
            if (epoch + 1) % val_interval == 0 or epoch == cf['num_epochs'] - 1:
                text_input = tokenizer(
                    prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
                    )

                with torch.no_grad():
                    eval_text_embeddings = text_encoder(text_input.input_ids.to(accelerator.device))[0]
                val_data = next(iter(val_dataloader))
                val_data = val_data[0]
                slo, first_ffa, second_ffa = val_data[0], val_data[1], val_data[2]
                latents_cond, latents_tar = pipeline(cf['eval_bc'], conditions=slo, text_emb=eval_text_embeddings)
                cond = autoencoderkl_tar.decode(latents_cond / scaling_factor, return_dict=False)[0]
                tar = autoencoderkl_tar.decode(latents_cond / scaling_factor, return_dict=False)[0]
                first_ffa = torch.cat([cond, first_ffa], dim=-1)
                second_ffa = torch.cat([tar, second_ffa], dim=-1)
                first_ffa = (make_grid(first_ffa, nrow=1).unsqueeze(0) + 1) / 2 
                second_ffa = (make_grid(second_ffa, nrow=1).unsqueeze(0) + 1) / 2 
                accelerator.trackers[0].log_images({'Early': first_ffa.cpu().detach(), 
                                                    'Late': second_ffa}, epoch+1) 
            
            if (epoch + 1) % save_interval == 0 or epoch == cf['num_epochs'] - 1:
                if cf['only_save_part']:
                    accelerator.unwrap_model(model).train_unet.save_pretrained(cf['project_dir']) 
                    noise_scheduler.save_pretrained(cf['project_dir']) 
                else:
                    pipeline.save_pretrained(cf['project_dir']) 
                
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config/control_attn.yaml')

    args = parser.parse_args()
    main(args) 
                
        