# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Optional, Union, Tuple, List, Callable, Dict
from IPython.display import display
# from tqdm.notebook import tqdm
from tqdm.auto import tqdm

## custom added(to support inputs_embeds directly to CLIPTextModel)
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
import torch
import torch.nn as nn
from typing import Optional

def CustomCLIPTextTransformerForward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutputWithPooling:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        if(inputs_embeds is not None):
            hidden_states = self.embeddings(inputs_embeds=inputs_embeds, position_ids=position_ids)
        else:
            hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )

        # expand attention_mask
        if attention_mask is not None and self.config._attn_implementation != "flash_attention_2":
            # [batch_size, seq_len] -> [batch_size, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        if self.eos_token_id == 2:
            # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
            # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
            # ------------------------------------------------------------
            # text_embeds.shape = [batch_size, sequence_length, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
            ]
        else:
            # The config gets updated `eos_token_id` from PR #24773 (so the use of extra new tokens is possible)
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
                # Note: we assume each sequence (along batch dim.) contains an  `eos_token_id` (e.g. prepared by the tokenizer)
                (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.eos_token_id)
                .int()
                .argmax(dim=-1),
            ]
        #print("CustomCLIPTextTransformerForward called")
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
def CustomCLIPTextModelForward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutputWithPooling:
        r"""
        Examples:

        ```python
        >>> from transformers import AutoTokenizer, CLIPTextModel

        >>> model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""
        #print("CustomCLIPTextModelForward called")
        return self.text_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )


# Function to replace the forward methods
# Enbable this function to replace the forward methods of CLIPTextModel and CLIPTextTransformer
# Support to pass inputs_embeds directly to CLIPTextModel
def replace_call_method_for_CLIPTextModel(model):
    model.forward = CustomCLIPTextModelForward.__get__(model)
    model.text_model.forward = CustomCLIPTextTransformerForward.__get__(model.text_model)
    return model




def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img


def view_images(images, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    display(pil_img)


def diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False,return_x0=False):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    
    res = model.scheduler.step(noise_pred, t, latents)
    latents = res.prev_sample
    x_0 = res.pred_original_sample
    
    latents = controller.step_callback(latents)
    if not return_x0:
        return latents
    else:
        return latents, x_0


@torch.no_grad()
def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    latents = latents.to(vae.dtype)
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    # 轉換成 float32 才能轉換成 numpy (bfloat16 不支援)
    image = image.float().cpu().permute(0, 2, 3, 1).detach().numpy()
    image = (image * 255).astype(np.uint8)
    return image#shape (batch, height, width, 3)
@torch.no_grad()
def image2latent(vae, image):
    #vae: AutoencoderKL
    #image: np.ndarray shape (batch, height, width, 3)
    
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(0, 3, 1, 2).to(vae.device)
    image = 2.0 * image - 1.0
    image= image.to(vae.dtype)
    latent = vae.encode(image).latent_dist.sample() * 0.18215
    return latent


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size,  model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


@torch.no_grad()
def text2image_ldm(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
):
    register_attention_control(model, controller)
    height = width = 256
    batch_size = len(prompt)
    
    uncond_input = model.tokenizer([""] * batch_size, padding="max_length", max_length=77, return_tensors="pt")
    uncond_embeddings = model.bert(uncond_input.input_ids.to(model.device))[0]
    
    text_input = model.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
    text_embeddings = model.bert(text_input.input_ids.to(model.device))[0]
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    context = torch.cat([uncond_embeddings, text_embeddings])
    
    model.scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale)
    
    image = latent2image(model.vqvae, latents)
   
    return image, latent


@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt: List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    low_resource: bool = False,
):
    register_attention_control(model, controller)
    height = width = 512
    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    
    context = [uncond_embeddings, text_embeddings]
    if not low_resource:
        context = torch.cat(context)
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    
    # set timesteps
    extra_set_kwargs = {}
    model.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource)
    
    image = latent2image(model.vae, latents)
  
    return image, latent


def get_module_name(model, target_module):
    for name, module in model.named_modules():
        if module is target_module:
            return name
    return None



def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = encoder_hidden_states is not None
            context = encoder_hidden_states if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q = self.head_to_batch_dim(q)
            k = self.head_to_batch_dim(k)
            v = self.head_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~attention_mask, max_neg_value)
            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            attn = controller(attn, is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.batch_to_head_dim(out)
            return to_out(out)

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet,name=None):
        if name == 'attn2':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                name__= get_module_name(net_,net__)
                count = register_recr(net__, count, place_in_unet,name__)
        return count


    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count
    print(f"Registered {cross_att_count} cross attention layers.")

    
def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def update_alpha_time_word(alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int,
                           word_inds: Optional[torch.Tensor]=None):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(prompts, num_steps,
                                   cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
                                   tokenizer, max_num_words=77):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"],
                                                  i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
             inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
             for i, ind in enumerate(inds):
                 if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
    return alpha_time_words


import os
def beta_weight(t, T, device="cuda", dtype=None):
    """
    計算時間步的權重: e^(t-T)
    
    Args:
        t: 當前步驟索引 (0, 1, 2, ..., T-1)
        T: 總步數
        device: 計算設備
        dtype: tensor 的 dtype，預設為 float32
    
    Returns:
        權重值
    """
    #to do 這邊先fix住，之後再調整權重
    ratio = t / T
    if dtype is None:
        dtype = torch.float32
    return torch.exp(torch.tensor(ratio-1, dtype=dtype, device=device))
@torch.no_grad()
def text2image_ldm_stable_with_learned_embedding(
    ldm_stable,
    learned_emb,
    controller,
    latent,
    num_inference_steps,
    guidance_scale: float = 7.5,
    low_resource: bool = False,
):
    register_attention_control(ldm_stable, controller)
    height = width = 512
    batch_size = 2

    uncond_input = ldm_stable.tokenizer(
        [""], padding="max_length", max_length=learned_emb.shape[1], return_tensors="pt"
    )
    uncond_embeddings = ldm_stable.text_encoder(uncond_input.input_ids.to(ldm_stable.device))[0]

    context = [uncond_embeddings,uncond_embeddings,uncond_embeddings, learned_emb]  # [unc,unc,text,text]
    if not low_resource:
        context = torch.cat(context)
    latent, latents = init_latent(latent, ldm_stable, height, width, None, batch_size)

    # set timesteps
    extra_set_kwargs = {}
    ldm_stable.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
    for t in tqdm(ldm_stable.scheduler.timesteps):
        latents = diffusion_step(ldm_stable, controller, latents, context, t, guidance_scale, low_resource)

    image = latent2image(ldm_stable.vae, latents)

    return image, latent





   
    



   


    

def train_text_embedding_ldm_stable(
    model,
    coarse_description: str,
    controller,
    latent: torch.FloatTensor,
    target_latent: torch.FloatTensor,
    num_steps: int = 50,
    epoch: int = 50,
    guidance_scale: float = 7.5,
    low_resource: bool = False,
    tau: float = 0.7,
    lr=1e-3,
    optimizer_cls=torch.optim.AdamW,
    save_interval: int = 5,
    save_image_dir: Optional[str] = None,
    beta_weighting: bool = False,
):
    register_attention_control(model, controller)
    model.text_encoder=replace_call_method_for_CLIPTextModel(model.text_encoder)
    os.makedirs(save_image_dir, exist_ok=True)
    height = width = 512


    #fix this with only train text embedding rather than text embedding(with text encoder)
    text_input = model.tokenizer(
        [coarse_description],
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""], padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]


    #before , after ->2 latents

    # set timesteps
    extra_set_kwargs = {}
    model.scheduler.set_timesteps(num_steps, **extra_set_kwargs)

    learned_emb = torch.nn.Parameter(text_embeddings.clone().detach().requires_grad_(True))
    optimizer = optimizer_cls([learned_emb], lr=lr)
    loss_fn = torch.nn.MSELoss()
    
    epoch_pbar = tqdm(range(epoch), desc="Training Epochs", leave=False)
    for ep in epoch_pbar:
        controller.reset()
        total_loss = []
        X_t=latent.expand(2, model.unet.in_channels, height // 8, width // 8).to(model.device)
        X_t = X_t.detach().clone()
        for i,t in enumerate(tqdm(model.scheduler.timesteps, desc="Diffusion Step", leave=False)):  # 0 to 49
            #reconstruct context with learned embedding
            cur_context = [uncond_embeddings,uncond_embeddings,uncond_embeddings, learned_emb]
            #concat
            cur_context = torch.cat(cur_context,dim=0)
            #uncond, uncond, uncond(before_image), text(after_image)
            X_t,x0 = diffusion_step(model, controller, X_t, cur_context, t, guidance_scale, low_resource,return_x0=True)
            X_t = X_t.detach()#no grad through 
            #compute loss(l2)
            if beta_weighting:
                loss= loss_fn(x0[1:], target_latent.expand_as(x0[1:])) *beta_weight(i,num_steps,device=model.device)
            else:
                loss= loss_fn(x0[1:], target_latent.expand_as(x0[1:]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
            #update loss to tqdm
            #clear cuda
            del x0
            torch.cuda.empty_cache()
        loss = sum(total_loss) / len(total_loss)
        # print(f"Epoch {ep + 1}, Loss: {loss:.4f}")
        epoch_pbar.set_postfix({'loss': f'{loss:.4f}'})     # 即時更新進度條旁邊的 Loss 數值，不用一直 Print
        if (ep) % save_interval == 0:
            image=latent2image(model.vae, X_t[1:].detach())
            PIL_image=Image.fromarray(image[0])
            PIL_image.save(os.path.join(save_image_dir, f"epoch_{ep}.png"))
            #save loss
            with open(os.path.join(save_image_dir, f"loss_{ep}.txt"), "a") as f:
                for i,l in enumerate(total_loss):
                    f.write(f"{i},{l}\n")
                
            #save learned embedding
            torch.save(learned_emb.detach().cpu(), os.path.join(save_image_dir, f"epoch_{ep}.pt"))
    #final learned embedding and image
    image=latent2image(model.vae, X_t[1:].detach())
    PIL_image=Image.fromarray(image[0])
    PIL_image.save(os.path.join(save_image_dir, f"final.png"))
    torch.save(learned_emb.detach().cpu(), os.path.join(save_image_dir, f"final.pt"))
    return learned_emb.detach()




def train_text_embedding_ldm_stable_with_out_encode(
    model,
    coarse_description: str,
    controller,
    latent: torch.FloatTensor,
    target_latent: torch.FloatTensor,
    num_steps: int = 50,
    epoch: int = 50,
    guidance_scale: float = 7.5,
    low_resource: bool = False,
    tau: float = 0.7,
    lr=1e-3,
    optimizer_cls=torch.optim.AdamW,
    save_interval: int = 5,
    save_image_dir: Optional[str] = None,
    beta_weighting: bool = False,
):
    register_attention_control(model, controller)
    model.text_encoder=replace_call_method_for_CLIPTextModel(model.text_encoder)
    os.makedirs(save_image_dir, exist_ok=True)
    height = width = 512

    latent=latent.to(model.dtype)
    target_latent=target_latent.to(model.dtype)

    tokens = model.tokenizer(
        coarse_description,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    embeds = model.text_encoder.get_input_embeddings()(tokens.input_ids.to(model.device))
    embeds.requires_grad = False
    y_len=len(model.tokenizer(coarse_description).input_ids) - 2
    learnable_embeds = embeds[:, 1:1+y_len, :].clone().detach()
    max_length = tokens.input_ids.shape[-1]

    uncond_input = model.tokenizer(
        [""], padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    

    #before , after ->2 latents

    # set timesteps
    extra_set_kwargs = {}
    model.scheduler.set_timesteps(num_steps, **extra_set_kwargs)

    learned_emb = torch.nn.Parameter(learnable_embeds.clone().detach().requires_grad_(True))
    optimizer = optimizer_cls([learned_emb], lr=lr)
    loss_fn = torch.nn.MSELoss()
    
    epoch_pbar = tqdm(range(epoch), desc="Training Epochs", leave=False)
    for ep in epoch_pbar:
        controller.reset()
        total_loss = []
        X_t=latent.expand(2, model.unet.in_channels, height // 8, width // 8).to(model.device)
        X_t = X_t.detach().clone()
        for i,t in enumerate(tqdm(model.scheduler.timesteps, desc="Diffusion Step", leave=False)):# 0 to 49
            #reconstruct context with learned embedding
            cond_embeddings=torch.cat([embeds[:, :1, :], learned_emb, embeds[:, 1+y_len:, :]], dim=1)
            #save cond_embeddings for debug
            # torch.save(cond_embeddings.detach().cpu(), os.path.join(save_image_dir, f"cond_emb_epoch{ep}_step{i}_before_encode.pt"))
            encoded_cond_embeddings=model.text_encoder(input_ids=tokens.input_ids.to(model.device),inputs_embeds=cond_embeddings)["last_hidden_state"]
            # torch.save(encoded_cond_embeddings.detach().cpu(), os.path.join(save_image_dir, f"cond_emb_epoch{ep}_step{i}_after_encode.pt"))
            cur_context = [uncond_embeddings,uncond_embeddings,uncond_embeddings, encoded_cond_embeddings]
            #concat
            cur_context = torch.cat(cur_context,dim=0)
            
            #uncond, uncond, uncond(before_image), text(after_image)
            X_t,x0 = diffusion_step(model, controller, X_t, cur_context, t, guidance_scale, low_resource,return_x0=True)
            X_t = X_t.detach()#no grad through 
            #compute loss(l2)
            if beta_weighting:
                loss= loss_fn(x0[1:], target_latent.expand_as(x0[1:])) *beta_weight(i,num_steps,device=model.device,dtype=x0.dtype)
            else:
                loss= loss_fn(x0[1:], target_latent.expand_as(x0[1:]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
            #update loss to tqdm
            #clear cuda
            del x0
            torch.cuda.empty_cache()
        loss = sum(total_loss) / len(total_loss)
        # print(f"Epoch {ep + 1}, Loss: {loss:.4f}")
        epoch_pbar.set_postfix({'loss': f'{loss:.4f}'})
        if (ep) % save_interval == 0:
            image=latent2image(model.vae, X_t[1:].detach())
            PIL_image=Image.fromarray(image[0])
            PIL_image.save(os.path.join(save_image_dir, f"epoch_{ep}.png"))
            #save loss
            with open(os.path.join(save_image_dir, f"loss_{ep}.txt"), "a") as f:
                for i,l in enumerate(total_loss):
                    f.write(f"{i},{l}\n")
                
            #save learned embedding
            torch.save(encoded_cond_embeddings.detach().cpu(), os.path.join(save_image_dir, f"epoch_{ep}.pt"))
    #final learned embedding and image
    image=latent2image(model.vae, X_t[1:].detach())
    PIL_image=Image.fromarray(image[0])
    PIL_image.save(os.path.join(save_image_dir, f"final.png"))
    torch.save(encoded_cond_embeddings.detach().cpu(), os.path.join(save_image_dir, f"final.pt"))
    return learned_emb.detach()



