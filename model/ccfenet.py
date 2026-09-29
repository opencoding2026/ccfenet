from typing import List, Optional, Tuple, Union

import os
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from model.IXC.modeling_internlm_xcomposer2 import InternLMXComposer2ForCausalLM
from model.IXC.modeling_internlm2 import InternLM2Model
from model.sam2.build_sam import build_sam2_hf
from model.sam2.utils.transforms import SAM2Transforms
from transformers import TextStreamer

from model.spatial_prior_navigator import SpatialPriorNavigator
from model.target_semantic_alignment import TokenPruneModule
from model.cross_validation_hub import CrossValidationHub
from model.spatial_gated_fusion import SpatialGatedFusion

try:
    from transformers.generation.streamers import BaseStreamer
except:  # noqa # pylint: disable=bare-except
    BaseStreamer = None


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,
    eps=1e-6,
):
    """Resize masks to prediction size before computing Dice loss."""
    if inputs.shape[-2:] != targets.shape[-2:]:
        targets = F.interpolate(
            targets.unsqueeze(1).float(),
            size=inputs.shape[-2:],
            mode="nearest",
        ).squeeze(1).to(dtype=inputs.dtype, device=inputs.device)
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """Resize masks to prediction size before computing BCE loss."""
    if inputs.shape[-2:] != targets.shape[-2:]:
        targets = F.interpolate(
            targets.unsqueeze(1).float(),
            size=inputs.shape[-2:],
            mode="nearest",
        ).squeeze(1).to(dtype=inputs.dtype, device=inputs.device)
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


class GeoPixelMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(GeoPixelMetaModel, self).__init__(config)
        self.config = config
        self.config.train_mask_decoder = getattr(self.config, "train_mask_decoder", kwargs.get("train_mask_decoder", False))
        self.config.out_dim = getattr(self.config, "out_dim", kwargs.get("out_dim", 256))
        self.vision_pretrained = kwargs.get("vision_pretrained", None)
        self.initialize_geopixel_modules(self.config)

    def initialize_geopixel_modules(self, config):
        self.visual_model = build_sam2_hf(self.vision_pretrained)

        self._transform = SAM2Transforms(
                    resolution=self.visual_model.image_size,
                    mask_threshold=0.0,
                    max_hole_area=0.0,
                    max_sprinkle_area=0.0,
                )
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]
        
        for param in self.visual_model.parameters():
            param.requires_grad = False

        if config.train_mask_decoder:
            self.visual_model.sam_mask_decoder.train()
            for param in self.visual_model.sam_mask_decoder.parameters():
                param.requires_grad = True

        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_projection_layers = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_projection_layers)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True

        self.spatial_prior_navigator = SpatialPriorNavigator(
            feat_dim=1024,
            seg_dim=out_dim,
            hidden_dim=64,
        )
        self.token_prune_module = TokenPruneModule(
            embed_dim=288,
            proj_dim=out_dim,
            num_heads=8,
        )
        self.cross_validation_hub = CrossValidationHub(
            hidden_dim=8,
            safe_threshold=0.85,
            prune_threshold=0.5,
        )
        self.spatial_gated_fusion = SpatialGatedFusion(
            main_dim=out_dim,
            spatial_dim=1024,
            target_size=64,
        )


class GeoPixelModel(GeoPixelMetaModel, InternLM2Model):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(GeoPixelModel, self).__init__(config, **kwargs)
        self.config.use_cache = False


class GeoPixelForCausalLM(InternLMXComposer2ForCausalLM):
    def __init__(self,config,**kwargs,):
        
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        self.spatial_loss_weight = kwargs.pop("spatial_loss_weight", 1.0)
        self.hub_loss_weight = kwargs.pop("hub_loss_weight", 1.0)
        self.seg_token_idx = kwargs.pop("seg_token_idx")

        super().__init__(config)
        self.model = GeoPixelModel(config, **kwargs)
        self.vocab_size = config.vocab_size
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @staticmethod
    def _align_gt_masks_to_pred_shape(gt_masks: torch.Tensor, pred_masks: torch.Tensor) -> torch.Tensor:
        """用最近邻插值对齐标注与预测的空间尺寸。"""
        if gt_masks.shape[-2:] == pred_masks.shape[-2:]:
            return gt_masks
        aligned = F.interpolate(
            gt_masks.unsqueeze(1).float(),
            size=pred_masks.shape[-2:],
            mode="nearest",
        ).squeeze(1)
        return aligned.to(dtype=pred_masks.dtype, device=pred_masks.device)

    def _build_pruning_h_seg_matrix(self, pred_embeddings: List[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """将逐目标 SEG 嵌入补齐为 [B, N_seg_max, C]，并标记有效目标。"""
        batch_size = len(pred_embeddings)
        max_targets = max((pe.shape[0] if pe.numel() > 0 else 0) for pe in pred_embeddings)
        max_targets = max(max_targets, 1)
        out_dim = self.model.config.out_dim

        h_seg_matrix = torch.zeros(batch_size, max_targets, out_dim, device=self.device)
        seg_valid_mask = torch.zeros(batch_size, max_targets, device=self.device, dtype=torch.bool)

        for batch_idx, pe in enumerate(pred_embeddings):
            if pe.numel() == 0:
                continue
            cur_targets = pe.shape[0]
            h_seg_matrix[batch_idx, :cur_targets] = pe
            seg_valid_mask[batch_idx, :cur_targets] = True

        return h_seg_matrix, seg_valid_mask

    def encode_g_img(self, image):
        """未生成 SEG 查询时使用原始 SAM2 图像编码。"""
        if image is None:
            return None
        if isinstance(image, str):
            _, ext = os.path.splitext(image)
            if ext.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp','.tif'}:
                image = Image.open(image)
                w, h = image.size
                _orig_hw = [(h, w)] 
            else:
                print ('Unknow input format', image)
                return None
        else:
            assert isinstance(image, torch.Tensor)
            _orig_hw = [image.shape[:2]]
        image = self.model._transform(image)
        image = image[None, ...].to(self.device)
        assert ( len(image.shape) == 4 and image.shape[1] == 3), f"image must be of size 1x3xHxW, got {image.shape}"
        features = self.get_visual_embs(image)   
        return features,_orig_hw

    def get_visual_embs(self, img_batch: torch.FloatTensor):
        """原始 SAM2 编码路径。"""
        with torch.no_grad():
            torch.cuda.empty_cache()
            img_batch = img_batch.to(self.device)
            batch_size = img_batch.shape[0]
            assert (
                len(img_batch.shape) == 4 and img_batch.shape[1] == 3
            ), f"grounding_img_batch must be of size Bx3xHxW, got {img_batch.shape}"
            backbone_out = self.model.visual_model.forward_image(img_batch)
            _, vision_feats, _, _ = self.model.visual_model._prepare_backbone_features(backbone_out)
            if self.model.visual_model.directly_add_no_mem_embed:
                vision_feats[-1] = vision_feats[-1] + self.model.visual_model.no_mem_embed
            feats = [
                feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
                for feat, feat_size in zip(vision_feats[::-1], self.model._bb_feat_sizes[::-1])
            ][::-1]
            features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        return features

    def get_visual_embs_with_prune(
        self,
        img_batch: torch.FloatTensor,
        h_seg: torch.Tensor,
        clip_feat: torch.Tensor,
        seg_valid_mask: torch.Tensor = None,
    ):
        """在 Hiera Stage 2/3 边界计算逐目标分数并调制后续视觉编码。"""
        img_batch = img_batch.to(self.device)
        batch_size = img_batch.shape[0]

        # 冻结 Stage 1/2；深层编码通过连续分数向增强模块回传梯度。
        with torch.no_grad():
            prune_feat, boundary_outputs = self.model.visual_model.image_encoder.forward_shallow(img_batch)

        spn_dtype = next(self.model.spatial_prior_navigator.parameters()).dtype
        clip_feat = clip_feat.to(dtype=spn_dtype)
        h_seg = h_seg.to(dtype=spn_dtype)
        S_spatial_per_seg, modulated_feat = self.model.spatial_prior_navigator(
            clip_feat, h_seg, seg_valid_mask=seg_valid_mask
        )
        token_prune_module = self.model.token_prune_module
        tpm_dtype = next(token_prune_module.parameters()).dtype
        prune_feat = prune_feat.to(dtype=tpm_dtype)
        h_seg_tpm = h_seg.to(dtype=tpm_dtype)
        S_prune_per_seg = token_prune_module(
            prune_feat, h_seg_tpm, seg_valid_mask=seg_valid_mask
        )
        target_h, target_w = prune_feat.shape[1], prune_feat.shape[2]
        S_spatial_token_per_seg = self.model.spatial_prior_navigator.get_token_scores(
            S_spatial_per_seg, target_h=target_h, target_w=target_w
        )
        # 语义含义：只要任意一个目标认为该位置重要，就保留该空间/语义证据。
        S_spatial = S_spatial_per_seg.max(dim=1).values
        S_spatial_token = S_spatial_token_per_seg.max(dim=1).values
        S_prune = S_prune_per_seg.max(dim=1).values

        prune_scores, keep_mask = self.model.cross_validation_hub(S_spatial_token, S_prune)

        deep_context = torch.enable_grad() if self.training else torch.no_grad()
        with deep_context:
            backbone_out = self.model.visual_model.image_encoder.forward_deep(
                prune_feat, boundary_outputs, prune_scores
            )
        # 与 SAM2.forward_image 保持相同的高分辨率特征处理。
        if self.model.visual_model.use_high_res_features_in_sam:
            backbone_out["backbone_fpn"][0] = self.model.visual_model.sam_mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0]
            )
            backbone_out["backbone_fpn"][1] = self.model.visual_model.sam_mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1]
            )

        _, vision_feats, _, _ = self.model.visual_model._prepare_backbone_features(backbone_out)
        if self.model.visual_model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.visual_model.no_mem_embed
        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self.model._bb_feat_sizes[::-1])
        ][::-1]
        features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

        sgf_dtype = next(self.model.spatial_gated_fusion.parameters()).dtype
        image_embed = features["image_embed"].to(dtype=sgf_dtype)
        modulated_feat = modulated_feat.to(dtype=sgf_dtype)
        features["image_embed"] = self.model.spatial_gated_fusion(image_embed, modulated_feat)
        return features, S_spatial, prune_scores, modulated_feat

    def get_clip_spatial_features(self, image_tensor):
        """提取去掉 CLS token 的 CLIP 特征，返回 [B, 1024, 40, 40]。"""
        with torch.no_grad():
            clip_input = F.interpolate(
                image_tensor.float(),
                size=(560, 560),
                mode="bicubic",
                align_corners=False,
            ).to(dtype=self.vit.vision_tower.dtype, device=self.device)
            
            outputs = self.vit.vision_tower(
                clip_input, output_hidden_states=True
            )
            clip_features = outputs.hidden_states[-1][:, 1:]
        
        clip_spatial = clip_features.reshape(-1, 40, 40, 1024).permute(0, 3, 1, 2)
        return clip_spatial
    
    def forward(self, **kwargs):
        return super().forward(**kwargs) if "past_key_values" in kwargs else self.model_forward(**kwargs)
    
    def model_forward(
            self,
            inference: bool = False,
            **kwargs,
    ):
        samples = kwargs.get('samples', None)
        if samples and samples['data_type'][0] == 'grounding':
            kwargs['output_hidden_states'] = True
            kwargs['use_cache'] = False

            torch.cuda.empty_cache()
            outputs = super().forward(**kwargs)

            if inference:
                assert len(samples['text_input']) == 1 and len(samples['image'][0]) == 1
                seg_token_mask = outputs.seg_token_mask
                output_hidden_states = [outputs.hidden_states]
                outputs = None
            else:
                output_hidden_states = outputs.hidden_states
                seg_token_mask = outputs.seg_token_mask

            hidden_states = []
            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))
            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = [states[masks] for states, masks in zip(last_hidden_state, seg_token_mask)]
            
            image_g_batch = torch.cat(samples['image_g'][0], dim=0)
            
            clip_feat = self.get_clip_spatial_features(image_g_batch)
            
            # 保留每个目标自己的 SEG embedding，后续分别计算再在目标维度做 max 融合。
            h_seg_matrix, seg_valid_mask = self._build_pruning_h_seg_matrix(pred_embeddings)
            
            image_g_features, S_spatial, prune_scores, modulated_feat = \
                self.get_visual_embs_with_prune(
                    image_g_batch, h_seg_matrix, clip_feat, seg_valid_mask=seg_valid_mask
                )
            
            ori_hw = samples['ori_hw'][0]
            all_pred_masks = []
            for i in range(len(pred_embeddings)):
                if (pred_embeddings[i].numel()== 0):
                    all_pred_masks.append([])
                    continue
                (sparse_embeddings, dense_embeddings,) = self.model.visual_model.sam_prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )
                batch_mode = (pred_embeddings[i].shape[0]>1)
                high_res_features = [
                    feat_level[i].unsqueeze(0)
                    for feat_level in image_g_features["high_res_feats"]
                ]
                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                image_g_embeds = image_g_features['image_embed'][i].unsqueeze(0).to(torch.bfloat16)
                low_res_masks, _, _ , _ = self.model.visual_model.sam_mask_decoder(
                    image_embeddings=image_g_embeds,
                    image_pe=self.model.visual_model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    repeat_image=batch_mode,
                    multimask_output=False,
                    high_res_features=high_res_features,
                )
                pred_masks = self.model._transform.postprocess_masks(
                    low_res_masks,
                    ori_hw[i],
                )
                all_pred_masks.append(pred_masks[:, 0])
                

            model_output = outputs
            gt_masks =  samples['masks'][0]
            pred_masks = all_pred_masks

            if inference:
                inference_outputs = {
                    "pred_masks": pred_masks,
                    "gt_masks": gt_masks,
                }
                return inference_outputs

            ce_loss = model_output.loss
            ce_loss = ce_loss * self.ce_loss_weight
            mask_bce_loss = 0
            mask_dice_loss = 0
            num_masks = 0

            for batch_idx in range(len(pred_masks)):
                cur_gt_masks = torch.stack(
                    [
                        torch.from_numpy(gt_mask).to(dtype=pred_masks[batch_idx].dtype, device=pred_masks[batch_idx].device)
                        for gt_mask in gt_masks[batch_idx]
                    ],
                    dim=0
                )
                cur_pred_masks = pred_masks[batch_idx]
                assert (
                    cur_gt_masks.shape[0] == cur_pred_masks.shape[0]
                ), "gt_masks.shape: {}, pred_masks.shape: {}".format(
                    cur_gt_masks.shape, cur_pred_masks.shape
                )
                cur_gt_masks = self._align_gt_masks_to_pred_shape(cur_gt_masks, cur_pred_masks)
                mask_bce_loss += (
                    sigmoid_ce_loss(cur_pred_masks, cur_gt_masks, num_masks=cur_gt_masks.shape[0])
                    * cur_gt_masks.shape[0]
                )
                mask_dice_loss += (
                    dice_loss(cur_pred_masks, cur_gt_masks, num_masks=cur_gt_masks.shape[0])
                    * cur_gt_masks.shape[0]
                )
                num_masks += cur_gt_masks.shape[0]

            mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
            mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
            mask_loss = mask_bce_loss + mask_dice_loss

            _prune_dtype = S_spatial.dtype if S_spatial is not None else torch.bfloat16
            spatial_loss = torch.tensor(0.0, device=self.device, dtype=_prune_dtype)
            hub_loss = torch.tensor(0.0, device=self.device, dtype=_prune_dtype)
            if S_spatial is not None and num_masks > 0:
                for batch_idx in range(len(gt_masks)):
                    cur_gt = torch.stack(
                        [torch.from_numpy(gm).to(dtype=_prune_dtype, device=self.device) for gm in gt_masks[batch_idx]],
                        dim=0
                    )
                    cur_gt = self._align_gt_masks_to_pred_shape(cur_gt, pred_masks[batch_idx])
                    # 对多目标真值取并集作为 SPN 和 CVH 的辅助监督。
                    gt_union = (cur_gt.sum(dim=0, keepdim=True) > 0).to(dtype=_prune_dtype).unsqueeze(0)
                    
                    spatial_loss += self.model.spatial_prior_navigator.compute_loss(
                        S_spatial[batch_idx:batch_idx+1], gt_union
                    )
                    
                    _n_tokens = prune_scores.shape[1]
                    _gt_side = int(_n_tokens ** 0.5)
                    gt_token_labels = self.model.spatial_prior_navigator.get_gt_token_labels(
                        gt_union, target_h=_gt_side, target_w=_gt_side
                    )
                    hub_loss += self.model.cross_validation_hub.compute_loss(
                        prune_scores[batch_idx:batch_idx+1], gt_token_labels
                    )
                
                spatial_loss = spatial_loss / len(gt_masks)
                hub_loss = hub_loss / len(gt_masks)

            spatial_loss = self.spatial_loss_weight * spatial_loss
            hub_loss = self.hub_loss_weight * hub_loss

            loss = ce_loss + mask_loss + spatial_loss + hub_loss
            outputs = CausalLMOutputWithPast(
                loss=loss,
                logits=model_output.logits,
                past_key_values=model_output.past_key_values,
                hidden_states=output_hidden_states,
                attentions=model_output.attentions,
            )
            outputs.ce_loss = ce_loss
            outputs.mask_bce_loss = mask_bce_loss
            outputs.mask_dice_loss = mask_dice_loss
            outputs.mask_loss = mask_loss
            outputs.spatial_loss = spatial_loss
            outputs.hub_loss = hub_loss
        else:
            outputs =  super().forward(**kwargs)
        return outputs

    def encode_g_img_with_prune(self, image, h_seg, seg_valid_mask=None):
        """返回增强后的 SAM2 特征与图像原始尺寸。"""
        if image is None:
            return None, None
        if isinstance(image, str):
            _, ext = os.path.splitext(image)
            if ext.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tif'}:
                image_pil = Image.open(image)
                w, h = image_pil.size
                _orig_hw = [(h, w)]
            else:
                print('Unknown input format', image)
                return None, None
        else:
            assert isinstance(image, torch.Tensor)
            _orig_hw = [image.shape[:2]]
            image_pil = image

        image_sam = self.model._transform(image_pil)
        image_sam = image_sam[None, ...].to(self.device)

        clip_feat = self.get_clip_spatial_features(image_sam)

        features, _, _, _ = self.get_visual_embs_with_prune(
            image_sam, h_seg, clip_feat, seg_valid_mask=seg_valid_mask
        )
        return features, _orig_hw

    def evaluate(
        self,
        tokenizer,
        query: str,
        images: List[Tuple[str, str]] = [],
        hd_num: int = 9,
        history: List[Tuple[str, str]] = [],
        max_new_tokens: int = 1024,
        stream: bool = False,
        **kwargs,
    ):
        with torch.no_grad():
            inputs, im_mask, _ = self.interleav_wrap_chat(query, images, history=history, hd_num=hd_num)
            inputs = {
                k: v.to(self.device)
                for k, v in inputs.items() if torch.is_tensor(v)
            }
            eos_token_id = [
                tokenizer.eos_token_id,
            ]
            all_pred_masks = []
            
            if stream:
                streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            else:
                streamer = None

            outputs = self.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                im_mask=im_mask,
                input_ids = None,
                streamer= streamer,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                top_p= 1.0,
                top_k = 0,
                eos_token_id=eos_token_id,
                repetition_penalty=1.0,
                infer_mode = 'base',
                output_hidden_states=True,
                return_dict_in_generate=True,
                **kwargs,
            )
            output_ids = outputs['sequences']
            response = tokenizer.decode(output_ids[0].cpu().tolist(), skip_special_tokens=True)
            response = response.replace("[UNUSED_TOKEN_145]","")
            history = history + [(query, response)]
            if len(images)==1 and isinstance(images[0], str):
                output_hidden_states = outputs.hidden_states[-1]
                seg_token_mask = output_ids[:, 1:-1] == self.seg_token_idx
                inputs_embeds_len = inputs['inputs_embeds'].size(1)
                seg_token_mask = torch.cat(
                    [
                        torch.zeros((seg_token_mask.shape[0], inputs_embeds_len)).bool().cuda(),
                        seg_token_mask,
                    ],
                    dim=1,
                )
                hidden_states = []
                assert len(self.model.text_hidden_fcs) == 1
                hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))
                last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
                pred_embeddings = [states[masks] for states, masks in zip(last_hidden_state, seg_token_mask)]
                
                # 保留所有生成的 SEG embedding，不在进入增强模块前做平均压缩。
                h_seg_matrix, seg_valid_mask = self._build_pruning_h_seg_matrix(pred_embeddings) if any(
                    pe.numel() > 0 for pe in pred_embeddings
                ) else (None, None)
                
                if h_seg_matrix is not None:
                    image_g_features, ori_hw = self.encode_g_img_with_prune(
                        images[0], h_seg_matrix, seg_valid_mask=seg_valid_mask
                    )
                else:
                    image_g_features, ori_hw = self.encode_g_img(images[0])

                for i in range(len(pred_embeddings)):
                    if (pred_embeddings[i].numel()== 0):
                        all_pred_masks.append([])
                        continue
                    (sparse_embeddings,dense_embeddings,) = self.model.visual_model.sam_prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                        text_embeds=pred_embeddings[i].unsqueeze(1),
                    )
                    batch_mode = (pred_embeddings[i].shape[0]>1)
                    high_res_features = [
                        feat_level[i].unsqueeze(0)
                        for feat_level in image_g_features["high_res_feats"]
                    ]
                    sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                    image_g_embeds = image_g_features['image_embed'][i].unsqueeze(0).to(torch.bfloat16)

                    low_res_masks, _, _ , _  = self.model.visual_model.sam_mask_decoder(
                        image_embeddings=image_g_embeds,
                        image_pe=self.model.visual_model.sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        repeat_image=batch_mode,
                        multimask_output=False,
                        high_res_features=high_res_features,
                    )
                    pred_masks = self.model._transform.postprocess_masks(
                        low_res_masks,
                        ori_hw[i],
                    )
                    all_pred_masks.append(pred_masks[:, 0])

        return response, all_pred_masks
