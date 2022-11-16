# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.utils import print_log

from mmseg.utils import get_root_logger
from mmseg.ops import resize
from .base import BaseModel
from seqtr.models import \
    (MODELS,
     build_vis_enc,
     build_lan_enc,
     build_fusion,
     build_head)


@MODELS.register_module()
class MaskClipPlus(BaseModel):

    def __init__(self, word_emb,num_token,vis_enc,lan_enc,head,fusion, text_categories,
                 text_channels, text_embeddings_path,
                    clip_weights_path=None, reset_counter=False, clip_channels=None, 
                    vit=False, **kwargs):
        super(MaskClipPlus, self).__init__()
        self.text_channels = text_channels
        self.text_embeddings_path = text_embeddings_path

        self.register_buffer('_iter_counter', torch.tensor(0, device='cuda'))
        self.clip_weights_path = clip_weights_path

        self.reset_counter = reset_counter
        if clip_channels is None:
            clip_channels = self.in_channels

        self.vis_enc = build_vis_enc(vis_enc)
        self.lan_enc = build_lan_enc(lan_enc, {'word_emb': word_emb,
                                               'num_token': num_token})
        self.decode_module = self.build_decode_module(head)

        del self.decode_module.loss_decode
        del self.decode_module.conv_seg
        del self.decode_module.dropout

        self.register_buffer('text_embeddings', torch.randn(text_categories, text_channels))

        self.vit = vit
        if vit:
            self.proj = nn.Conv2d(clip_channels, text_channels, 1, bias=False)
        else:
            self.q_proj = nn.Conv2d(clip_channels, clip_channels, 1)
            self.k_proj = nn.Conv2d(clip_channels, clip_channels, 1)
            self.v_proj = nn.Conv2d(clip_channels, clip_channels, 1)
            self.c_proj = nn.Conv2d(clip_channels, text_channels, 1)
        super(MaskClipPlus, self).init_weights()

    def build_decode_module(self, cfg):
        cfg['init_cfg'] = None
        cfg['in_channels'] = self.in_channels
        cfg['channels'] = self.channels
        self.decode_module = build_head(cfg)
        del self.decode_module.loss_decode
        del self.decode_module.conv_seg
        del self.decode_module.dropout
        
    def init_weights(self):
        self.load_text_embeddings()
        self.load_clip_weights()

    def load_text_embeddings(self):
        loaded = torch.load(self.text_embeddings_path, map_location='cuda')
        self.text_embeddings[:, :] = loaded[:, :]
        print_log(f'Loaded text embeddings from {self.text_embeddings_path}', logger=get_root_logger())

    def load_clip_weights(self):
        loaded = torch.load(self.clip_weights_path, map_location='cuda')
        self.vis_enc.load_state_dict(loaded['clip'])
        attrs = ['proj'] if self.vit else ['q_proj', 'k_proj', 'v_proj', 'c_proj']
        for attr in attrs:
            current_attr = getattr(self, attr)
            state_dict = loaded[attr]
            for key in state_dict:
                if 'weight' in key:
                    state_dict[key] = state_dict[key][:, :, None, None]
            current_attr.load_state_dict(state_dict)
        print_log(f'Loaded clip weights from {self.clip_weights_path}', logger=get_root_logger())

    def _freeze(self):
        """Freeze params and norm stats."""
        super(MaskClipPlus, self)._freeze()
        attrs = ['proj'] if self.vit else ['q_proj', 'k_proj', 'v_proj', 'c_proj']
        attrs.append('clip')
        for attr in attrs:
            i = getattr(self, attr)
            for m in i.modules():
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward_train(self,img,
                      ref_expr_inds,
                      img_metas,
                      gt_mask=None,
                      rescale=False):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.vis_enc(img)[-1]
        q, k, v, cls_token = None, None, None, None
        if self.vit:
            if isinstance(x, list) and len(x) == 4:
                x, q, k, v = x
            if isinstance(x, list) and len(x) == 2:
                x, cls_token = x
            if v is not None:
                feat = self.proj(v)
            else:
                feat = self.proj(x)
            if cls_token is not None:
                cls_token = self.proj(cls_token[:, :, None, None])[:, :, 0, 0]
        else:
            q = self.q_proj(x)
            k = self.k_proj(x)
            q = torch.flatten(q, start_dim=2).transpose(-2, -1)
            k = torch.flatten(k, start_dim=2).transpose(-2, -1)
            v = self.v_proj(x)
            feat = self.c_proj(v)

        feat = feat / feat.norm(dim=1, keepdim=True)
        gt_mask = gt_mask.squeeze(1)

        output = torch.einsum('nchw,lc->nlhw', [feat, self.text_embeddings])

        output = resize(
            input=output,
            size=gt_mask.shape[1:],
            mode='bilinear',
            align_corners=self.align_corners)
                
        losses = self.losses(output, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs)[0]