# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='MaskClipPlus',
    pretrained='pretrain/ViT16_clip_weights',
    vis_enc=dict(
        type='VisionTransformerMaskClip',
        img_size=(224, 224),
        patch_size=16,
        patch_bias=False,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        out_indices=-1,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        with_cls_token=True,
        output_cls_token=False,
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        patch_norm=False,
        pre_norm = True,
        final_norm=True,
        return_qkv=True,
        interpolate_mode='bicubic',
        num_fcs=2,
        norm_eval=False
    ),
    
    fusion=dict(
        type='ClipFusion'
    ),
    
    lan_enc=dict(
        type='CLIPTextEncoder',
        context_length=17, # if refcocog 22
        embed_dim=512, # if RN50 1024, if RN101, vit 512
        pretrained=None
    ),
    
    head=dict(
        type='ASPPHeadV2',
        input_transform=None,
        dilations=(6, 12, 18, 24)
    ),
    vit=True,
    in_channels=768,
    channels=512,
    num_classes=20,
    dropout_ratio=0,
    norm_cfg=norm_cfg,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
    ),
    decode_module_cfg=dict(
        type='ASPPHeadV2',
        input_transform=None,
        dilations=(6, 12, 18, 24)
    ),
    text_categories=20,
    text_channels=512,
    clip_channels=768,
    text_embeddings_path='pretrain/voc_ViT16_clip_text.pth',
    cls_bg=False,
    norm_feat=False,
    clip_unlabeled_cats=list(range(0, 20)),
    clip_cfg=dict(
        type='VisionTransformerMaskClip',
        img_size=(224, 224),
        patch_size=16,
        patch_bias=False,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        out_indices=-1,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        with_cls_token=True,
        output_cls_token=False,
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        patch_norm=False,
        pre_norm = True,
        final_norm=True,
        return_qkv=True,
        interpolate_mode='bicubic',
        num_fcs=2,
        norm_eval=False
    ),
    clip_weights_path='pretrain/ViT16_clip_weights.pth',
    reset_counter=True,
    start_clip_guided=(1, -1),
    start_self_train=(-1, -1),
    feed_img_to_decode_head=True,
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)
