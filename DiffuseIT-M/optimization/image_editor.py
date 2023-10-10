import os
from pathlib import Path
from optimization.constants import ASSETS_DIR_NAME, RANKED_RESULTS_DIR

from utils_visualize.metrics_accumulator import MetricsAccumulator
from utils_visualize.video import save_video

from numpy import random
from optimization.augmentations import ImageAugmentations

from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import functional as TF
from torch.nn.functional import mse_loss
from optimization.losses import range_loss, d_clip_loss
# import lpips
import numpy as np
from src.vqc_core import *
from model_vit.loss_vit import Loss_vit
from CLIP import clip
from guided_diffusion.guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from utils_visualize.visualization import show_tensor_image, show_editied_masked_image
from pathlib import Path


from color_matcher import ColorMatcher
from color_matcher.io_handler import load_img_file, save_img_file, FILE_EXTS
from color_matcher.normalizer import Normalizer
import lpips
mean_sig = lambda x:sum(x)/len(x)
class ImageEditor:
    def __init__(self, args) -> None:
        self.args = args
        os.makedirs(self.args.output_path, exist_ok=True)

        self.ranked_results_path = Path(self.args.output_path)
        os.makedirs(self.ranked_results_path, exist_ok=True)

        if self.args.seed is not None:
            torch.manual_seed(self.args.seed)
            np.random.seed(self.args.seed)
            random.seed(self.args.seed)

        self.model_config = model_and_diffusion_defaults()
        if self.args.use_ffhq:
            self.model_config.update(
            {
                "attention_resolutions": "16",
                "class_cond": self.args.model_output_size == 512,
                "diffusion_steps": 1000,
                "rescale_timesteps": True,
                "timestep_respacing": self.args.timestep_respacing,
                "image_size": self.args.model_output_size,
                "learn_sigma": True,
                "noise_schedule": "linear",
                "num_channels": 128,
                "num_head_channels": 64,
                "num_res_blocks": 1,
                "resblock_updown": True,
                "use_fp16": False,
                "use_scale_shift_norm": True,
            }
        )
        else:
            self.model_config.update(
            {
                "attention_resolutions": "32, 16, 8",
                "class_cond": self.args.model_output_size == 512,
                "diffusion_steps": 1000,
                "rescale_timesteps": True,
                "timestep_respacing": self.args.timestep_respacing,
                "image_size": self.args.model_output_size,
                "learn_sigma": True,
                "noise_schedule": "linear",
                "num_channels": 256,
                "num_head_channels": 64,
                "num_res_blocks": 2,
                "resblock_updown": True,
                "use_fp16": True,
                "use_scale_shift_norm": True,
            }
        )

        # Load models
        self.device = torch.device(
            f"cuda:{self.args.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        print("Using device:", self.device)

        self.model, self.diffusion = create_model_and_diffusion(**self.model_config)

        self.model.load_state_dict(
        torch.load(
            "./checkpoints/256x256_diffusion_uncond.pt"
            if self.args.model_output_size == 256
            else "checkpoints/512x512_diffusion.pt",
            map_location="cpu",
            )
        )
        self.model.requires_grad_(False).eval().to(self.device)
        for name, param in self.model.named_parameters():
            if "qkv" in name or "norm" in name or "proj" in name:
                param.requires_grad_()
        if self.model_config["use_fp16"]:
            self.model.convert_to_fp16()
        with open("model_vit/config.yaml", "r") as ff:
            config = yaml.safe_load(ff)

        cfg = config

        self.VIT_LOSS = Loss_vit(cfg, lambda_ssim=self.args.lambda_ssim,lambda_dir_cls=self.args.lambda_dir_cls,lambda_contra_ssim=self.args.lambda_contra_ssim,lambda_trg=args.lambda_trg).eval()#.requires_grad_(False)
        self.lpips_model = lpips.LPIPS(net="vgg").to(self.device)

        # init networks


        self.cm = ColorMatcher()


        self.metrics_accumulator = MetricsAccumulator()

        self.image_size = (self.model_config["image_size"], self.model_config["image_size"])
        self.args.init_mask = self.args.init_mask.replace('train_img','reverse_seg_train')
        self.mask_pil = Image.open(self.args.init_mask).convert('RGB')
        self.mask_pil = self.mask_pil.resize(self.image_size,Image.LANCZOS)
        image_mask_pil_binarized = ((np.array(self.mask_pil) > 10) * 255).astype(np.uint8)
        self.mask = TF.to_tensor(Image.fromarray(image_mask_pil_binarized))
        self.mask = self.mask.unsqueeze(0).to(self.device)

    def noisy_aug(self,t,x,x_hat):
        fac = self.diffusion.sqrt_one_minus_alphas_cumprod[t]
        x_mix = x_hat * fac + x * (1 - fac)
        return x_mix

    def unscale_timestep(self, t):
        unscaled_timestep = (t * (self.diffusion.num_timesteps / 1000)).long()

        return unscaled_timestep

    def edit_image_by_prompt(self):

        self.image_size = (self.model_config["image_size"], self.model_config["image_size"])
        self.init_image_pil = Image.open(self.args.init_image).convert("RGB")
        self.init_image_pil = self.init_image_pil.resize(self.image_size, Image.LANCZOS)  # type: ignore
        self.init_image = (
            TF.to_tensor(self.init_image_pil).to(self.device).unsqueeze(0).mul(2).sub(1)
        )

        self.target_image = None
        if self.args.target_image is not None:
            self.target_image_pil = Image.open(self.args.target_image).convert("RGB")
            self.target_image_pil = self.target_image_pil.resize(self.image_size, Image.LANCZOS)  # type: ignore
            self.target_image = (
                TF.to_tensor(self.target_image_pil).to(self.device).unsqueeze(0).mul(2).sub(1)
            )


        self.prev = self.init_image.detach()
        self.flag_resample=False
        total_steps = self.diffusion.num_timesteps - self.args.skip_timesteps - 1
        def cond_fn(x, t, y=None):
            self.flag_resample=False
            with torch.enable_grad():
                frac_cont=1.0
                if self.target_image is None:
                    if self.args.use_prog_contrast:
                        if self.loss_prev > -0.5:
                            frac_cont = 0.5
                        elif self.loss_prev > -0.4:
                            frac_cont = 0.25
                    if self.args.regularize_content:
                        if self.loss_prev < -0.5:
                            frac_cont = 2
                x = x.detach().requires_grad_()
                t = self.unscale_timestep(t)

                out = self.diffusion.p_mean_variance(
                    self.model, x, t, clip_denoised=False, model_kwargs={"y": y}
                )

                loss = torch.tensor(0)
                if self.args.use_noise_aug_all:
                    x_in = self.noisy_aug(t[0].item(),x,out["pred_xstart"])
                else:
                    x_in = out["pred_xstart"]

                if self.args.vit_lambda != 0:

                    if t[0]>self.args.diff_iter:
                        vit_loss,vit_loss_val = self.VIT_LOSS(x_in,self.init_image,self.prev,use_dir=True,frac_cont=frac_cont,target = self.target_image)
                    else:
                        vit_loss,vit_loss_val = self.VIT_LOSS(x_in,self.init_image,self.prev,use_dir=False,frac_cont=frac_cont,target = self.target_image)
                    loss = loss + vit_loss

                if self.args.range_lambda != 0:
                    r_loss = range_loss(out["pred_xstart"]).sum() * self.args.range_lambda
                    loss = loss + r_loss
                    self.metrics_accumulator.update_metric("range_loss", r_loss.item())
                if self.target_image is not None:
                    loss = loss + mse_loss( x_in, self.target_image) * self.args.l2_trg_lambda

                self.prev = x_in.detach().clone()
                if True:# self.args.background_preservation_loss:
                    if self.mask is not None:
                        masked_background = x_in * (1 - self.mask)
                    else:
                        masked_background = x_in

                    if self.args.lpips_sim_lambda: # 1000
                        lpips_loss = self.lpips_model(masked_background, self.init_image).sum() * self.args.lpips_sim_lambda
                        loss = (
                            loss
                            + lpips_loss
                        )
                    if self.args.l2_trg_lambda:
                        loss = (
                            loss
                            + mse_loss(masked_background, self.init_image) * self.args.l2_trg_lambda
                        )
                    self.metrics_accumulator.update_metric("lpips",lpips_loss.item())

                if self.args.use_range_restart:
                    if t[0].item() < total_steps:
                        if self.args.use_ffhq:
                            if r_loss>0.1:
                                self.flag_resample =True
                        else:
                            if r_loss>0.01:
                                self.flag_resample =True

            return -torch.autograd.grad(loss, x)[0], self.flag_resample
        @torch.no_grad()
        def postprocess_fn(out, t):
            if self.mask is not None:
                background_stage_t = self.diffusion.q_sample(self.init_image, t[0])
                background_stage_t = torch.tile(
                    background_stage_t, dims=(self.args.batch_size, 1, 1, 1)
                )
                out["sample"] = out["sample"] * self.mask + background_stage_t * (1 - self.mask)

            return out

        save_image_interval = self.diffusion.num_timesteps // 5
        for iteration_number in range(self.args.iterations_num):
            print(f"Start iterations {iteration_number}")

            sample_func = (
                self.diffusion.ddim_sample_loop_progressive
                if self.args.ddim
                else self.diffusion.p_sample_loop_progressive
            )
            samples = sample_func(
                self.model,
                (
                    self.args.batch_size,
                    3,
                    self.model_config["image_size"],
                    self.model_config["image_size"],
                ),
                clip_denoised=False,
                model_kwargs={}
                if self.args.model_output_size == 256
                else {
                    "y": torch.zeros([self.args.batch_size], device=self.device, dtype=torch.long)
                },
                cond_fn=cond_fn,
                progress=True,
                skip_timesteps=self.args.skip_timesteps,
                init_image=self.init_image,
                postprocess_fn=postprocess_fn,
                randomize_class=True,
            )
            if self.flag_resample:
                continue

            intermediate_samples = [[] for i in range(self.args.batch_size)]
            total_steps = self.diffusion.num_timesteps - self.args.skip_timesteps - 1
            total_steps_with_resample= self.diffusion.num_timesteps - self.args.skip_timesteps - 1 + (self.args.resample_num-1)
            for j, sample in enumerate(samples):
                should_save_image = j % save_image_interval == 0 or j == total_steps_with_resample

                # self.metrics_accumulator.print_average_metric()

                for b in range(self.args.batch_size):
                    pred_image = sample["pred_xstart"][b]
                    visualization_path = Path(
                        os.path.join(self.args.output_path, self.args.output_file)
                    )
                    visualization_path = visualization_path.with_name(
                        f"{visualization_path.stem}_i_{iteration_number}_b_{b}{visualization_path.suffix}"
                    )
                    pred_image = (
                                self.init_image[0] * (1 - self.mask[0]) + pred_image * self.mask[0]
                            )
                    pred_image = pred_image.add(1).div(2).clamp(0, 1)
                    pred_image_pil = TF.to_pil_image(pred_image)
            ranked_pred_path = self.ranked_results_path / (visualization_path.name)
            if self.args.target_image is not None:
                if self.args.use_colormatch:
                    src_image = Normalizer(np.asarray(pred_image_pil)).type_norm()
                    trg_image = Normalizer(np.asarray(self.target_image_pil)).type_norm()
                    img_res = self.cm.transfer(src=src_image, ref=trg_image, method='mkl')
                    img_res = Normalizer(img_res).uint8_norm()
                    img_res_pil = Image.fromarray(img_res)
                    img_res_pil.save(ranked_pred_path)
                    save_img_file(img_res, str(ranked_pred_path))
            else:
                pred_image_pil.save(ranked_pred_path)


