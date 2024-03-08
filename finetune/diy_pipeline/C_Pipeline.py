from typing import Any
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
import torch
try:
    from diffusers.utils import randn_tensor
except:
    from diffusers.utils.torch_utils import randn_tensor
from typing import List, Optional, Tuple, Union

class REF_Pip_Tensor(DiffusionPipeline):
    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        class_labels = None, 
        conditions = None, 
        text_emb = None, 
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """
        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        
        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            if class_labels == None:
                model_output = self.unet(image, t, 
                                        text_emb, cross_attention_kwargs={'cond_lat': conditions}).sample
            else:
                model_output = self.unet(image, t, text_emb, 
                                         class_labels=class_labels, cross_attention_kwargs={'cond_lat': conditions}).sample

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        image = image.clamp(-1, 1)
        # image = image.cpu().numpy()
        
        return image

class Ctrl_Ref_Tensor(DiffusionPipeline):
    def __init__(self, unets, scheduler):
        super().__init__()
        self.register_modules(unet=unets, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        class_labels = None, 
        conditions = None, # conditions will be entered into the control model
        text_emb = None, 
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """
        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        
        first_pic = second_pic = randn_tensor(image_shape, generator=generator, device=self.unet.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        
        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            if class_labels == None:
                target_output, mid_output = self.unet(second_pic, first_pic, t, text_emb, 
                                        text_emb, conditions, return_mid=True)
            else:
                target_output, mid_output = self.unet(second_pic, first_pic, t, text_emb, 
                                        text_emb, conditions, return_mid=True)

            # 2. compute previous image: x_t -> x_t-1
            second_pic = self.scheduler.step(target_output, t, second_pic, generator=generator).prev_sample
            first_pic = self.scheduler.step(mid_output, t, first_pic, generator=generator).prev_sample

        first_pic = first_pic.clamp(-1, 1)
        second_pic = second_pic.clamp(-1, 1)
        
        
        return first_pic, second_pic

class C_Pipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        
    @torch.no_grad()
    def __call__(self, conditions, 
                 cond_add_noise=False, 
                 text_emb=None, 
                 class_labels=None) -> torch.tensor:
        self.unet.eval()
        conditions = conditions.to(self.unet.device)
        init_noise = torch.randn((conditions.shape[0], self.unet.config.in_channels//2, 
                              self.unet.config.sample_size, 
                              self.unet.config.sample_size), 
                              device=self.unet.device)
        cat = torch.cat([init_noise, conditions], dim=1)
        for t in self.scheduler.timesteps:
            if cond_add_noise:
                cache_conditions = cat[:, self.unet.config.in_channels//2:]
                timesteps = torch.tensor(t, device=init_noise.device).long().repeat(conditions.shape[0])
                cache_conditions = self.scheduler.add_noise(cache_conditions, init_noise, timesteps)
                cat[:, self.unet.config.in_channels//2:] = cache_conditions
            with torch.no_grad():
                if text_emb is None:
                    model_output = self.unet(cat, t, class_labels=class_labels).sample
                else:
                    model_output =  self.unet(cat, t, text_emb, class_labels=class_labels).sample
                image = self.scheduler.step(model_output, t, cat).prev_sample
            new_noise = image[:,:self.unet.config.in_channels//2]
            cat = torch.cat([new_noise, conditions], dim=1)
        result = cat[:,:self.unet.config.in_channels//2]
        self.unet.train()
        return result

