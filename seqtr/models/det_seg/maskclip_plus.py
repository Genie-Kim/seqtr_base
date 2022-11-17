# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.utils import print_log
from mmcv.runner import force_fp32
from mmseg.utils import get_root_logger
from mmseg.ops import resize
from .base import BaseModel
import pycocotools.mask as maskUtils
from seqtr.models import \
    (MODELS,
     build_vis_enc,
     build_lan_enc,
     build_fusion,
     build_head)
from mmseg.models.builder import build_loss
from mmseg.models.losses import Accuracy
@MODELS.register_module()
class MaskClipPlus(BaseModel):

    def __init__(self,vis_enc,lan_enc,head, text_channels,
                    clip_weights_path=None, reset_counter=False, clip_channels=None, 
                    vit=False,freeze_proj_vit=False,loss_decode = None,threshold=0.3, **kwargs):
        super(MaskClipPlus, self).__init__()
        self.text_channels = text_channels

        self.register_buffer('_iter_counter', torch.tensor(0, device='cuda'))
        self.clip_weights_path = clip_weights_path

        self.reset_counter = reset_counter
        if clip_channels is None:
            clip_channels = self.in_channels

        self.vis_enc = build_vis_enc(vis_enc)
        self.lan_enc = build_lan_enc(lan_enc)
        self.decode_module = build_head(head)
        self.vit = vit
        if vit:
            self.proj = nn.Conv2d(clip_channels, text_channels, 1, bias=False)
        else:
            self.q_proj = nn.Conv2d(clip_channels, clip_channels, 1)
            self.k_proj = nn.Conv2d(clip_channels, clip_channels, 1)
            self.v_proj = nn.Conv2d(clip_channels, clip_channels, 1)
            self.c_proj = nn.Conv2d(clip_channels, text_channels, 1)
            
        self.lan_train = lan_enc.do_train
        super(MaskClipPlus, self).init_weights()
        del self.decode_module.loss_decode
        # because we don't use Basedecoder's cls_seg funciton, this doesn't matter.        
        del self.decode_module.conv_seg
        del self.decode_module.dropout
        
        self.align_corners = False
        if isinstance(loss_decode, dict):
            self.loss_decode = build_loss(loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(build_loss(loss))
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict,\
                but got {type(loss_decode)}')
        self.ignore_index = 255
        self.threshold = threshold        
        
        if freeze_proj_vit:
            self._freeze()
       
    def init_weights(self):
        # self.load_text_embeddings()
        self.load_clip_weights()

    # def load_text_embeddings(self):
    #     loaded = torch.load(self.text_embeddings_path, map_location='cuda')
    #     self.text_embeddings[:, :] = loaded[:, :]
    #     print_log(f'Loaded text embeddings from {self.text_embeddings_path}', logger=get_root_logger())
        
        
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
        # super(MaskClipPlus, self)._freeze()
        attrs = ['proj'] if self.vit else ['q_proj', 'k_proj', 'v_proj', 'c_proj']
        attrs.append('clip')
        for attr in attrs:
            i = getattr(self, attr)
            for m in i.modules():
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward_train(self,img,
                      ref_expr,
                      img_metas,
                      gt_mask=None,
                      rescale=True):
        """Args:
            img (tensor): [batch_size, c, h_batch, w_batch].

            ref_expr_inds (tensor): [batch_size, max_token].

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `seqtr/datasets/pipelines/formatting.py:CollectData`.

            gt_mask_vertices (list[tensor]): [batch_size, 2, num_ray], padded values are -1, 
                the coordinates are in 'pad_shape' scale.

            rescale (bool): whether to rescale predictions from `img_shape`/`pad_shape`
                back to `ori_shape`.

        """
        
        # self.register_buffer('text_embeddings', torch.randn(text_categories, text_channels))
        # self.text_embeddings[:, :] = loaded[:, :]
        
        # extract image features.
        feat = self.extract_image_feature(img)  
        feat = self.decode_module.forward_module(feat) # features after decoder, before classifier
        feat = feat / feat.norm(dim=1, keepdim=True) # normalize
        
        # gt_mask = gt_mask.squeeze(1)
        if len(ref_expr.shape) == 3:
            ref_expr = ref_expr.squeeze(1)
            
        if self.lan_train:
            # dim(self.lan_enc(ref_expr)) == batch, transformerwidth
            text_emb = self.lan_enc(ref_expr).unsqueeze(1)
            output = torch.einsum('nchw,nlc->nlhw', [feat, text_emb])
        else:
            output = torch.einsum('nchw,lc->nlhw', [feat, img_metas['pseudo_text_emb']])

        seg_logits = resize(
            input=output,
            size=img.shape[-2:],
            mode='bilinear',
            align_corners=self.align_corners)
        
        losses_dict = self.losses(seg_logits, gt_mask)
              
        with torch.no_grad():
            predictions = self.get_predictions(seg_logits, img_metas, rescale=rescale)
        
        return losses_dict, predictions
    
    @torch.no_grad()
    def forward_test(self,
                     img,
                     ref_expr,
                     img_metas,
                     with_bbox=False,
                     with_mask=False,
                     rescale=True):
        """Args:
            img (tensor): [batch_size, c, h_batch, w_batch].

            ref_expr_inds (tensor): [batch_size, max_token], padded value is 0.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `rec/datasets/pipelines/formatting.py:CollectData`.

            with_bbox/with_mask: whether to generate bbox coordinates or mask contour vertices,
                which has slight differences.

            rescale (bool): whether to rescale predictions from `img_shape`/`pad_shape`
                back to `ori_shape`.
        """
        assert with_bbox==False and with_mask==True
        # extract image features.
        feat = self.extract_image_feature(img)  
        feat = self.decode_module.forward_module(feat) # features after decoder, before classifier
        feat = feat / feat.norm(dim=1, keepdim=True) # normalize
        
        if len(ref_expr.shape) == 3:
            ref_expr = ref_expr.squeeze(1)
            
        text_emb = self.lan_enc(ref_expr).unsqueeze(1)
        output = torch.einsum('nchw,nlc->nlhw', [feat, text_emb])
                
        seg_logits = resize(
            input=output,
            size=img.shape[-2:],
            mode='bilinear',
            align_corners=self.align_corners)
        
        predictions = self.get_predictions(seg_logits, img_metas, rescale=rescale)
        
        return predictions
     
    def extract_image_feature(self,img):
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
        return feat
    
    def get_predictions(self, seg_logits, img_metas, rescale=True):
        """
        Args:
            seg_logit (Tensor): The input image of shape (N, # of classes, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """
        pred_masks = list()
        pred_masks_notrle = list()
        for seg_logit, img_meta in zip(seg_logits, img_metas):
            
            output = torch.sigmoid(seg_logit)
            # output = F.softmax(seg_logit, dim=1)
            # flip = img_meta[0]['flip']
            # if flip:
            #     flip_direction = img_meta[0]['flip_direction']
            #     assert flip_direction in ['horizontal', 'vertical']
            #     if flip_direction == 'horizontal':
            #         output = output.flip(dims=(3, ))
            #     elif flip_direction == 'vertical':
            #         output = output.flip(dims=(2, ))
                    
            seg_pred = (output > self.threshold).to(output).squeeze(1)
            # seg_pred = output.argmax(dim=1)
            # if torch.onnx.is_in_onnx_export():
            #     # our inference backend only support 4D output
            #     seg_pred = seg_pred.unsqueeze(0)
            #     return seg_pred
            
            if rescale:
                seg_pred = F.interpolate(seg_pred.unsqueeze(0), img_meta['ori_shape'][:2], mode='bilinear', align_corners=False).squeeze()
            else:
                seg_pred = F.interpolate(seg_pred.unsqueeze(0), img_meta['pad_shape'][:2], mode='bilinear', align_corners=False).squeeze()
            
            pred_mask = seg_pred.cpu().numpy().astype(np.uint8)
            pred_masks_notrle.append(pred_mask)
            pred_mask = np.asfortranarray(pred_mask)
            pred_rle = maskUtils.encode(pred_mask)  # dict
            pred_masks.append(pred_rle)
        
        return dict(pred_masks = pred_masks,pred_masks_notrle=pred_masks_notrle)
    
    
    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        # seg_logit = resize(
        #     input=seg_logit,
        #     size=seg_label.shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        seg_weight = None
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

        # loss['acc_seg'] = self.accuracy(seg_logit, seg_label)
        return loss