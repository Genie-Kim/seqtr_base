# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='MaskClipPlus',
    vit=True,
    dropout_ratio=0,
    text_categories=1,
    text_channels=512,
    clip_channels=768,
    clip_weights_path='pretrain/ViT16_clip_weights.pth',
    reset_counter=True,
    loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0,avg_non_ignore=True),
    threshold=0.3, # if None, default 0.3   
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
        norm_eval=False,
        freeze_layer = None, # ex 3 --> freeze last 3 transformer layers.
        do_train = True
    ),
     
    lan_enc=dict(
        type='CLIPTextEncoder',
        context_length=17, # if refcocog 22
        pretrained='pretrain/ViT-B-16.pt',
        embed_dim= 512, # if RN50 1024, if RN101, vit 512
        do_train = True
    ),
    
    head=dict(
        type='ASPPHeadV2',
        input_transform=None,
        dilations=(6, 12, 18, 24),
        in_channels=512,
        channels=512,
        norm_cfg=norm_cfg,        
        residual=True,
        num_classes =1,
        # out_channels=1, # should 1
    ),
)

# ToDo: decoder part learning rate should be 1e-4
optimizer_config = dict(
    type='AdamW',
    lr=1e-4, # this brings from CRIS(CVPR'22) --> decode head lr
    betas=(0.9, 0.98), # for vit
    eps=1e-6, # for vit
    # betas=(0.9, 0.999), # for resnet
    # eps=1e-8, # for resnet
    weight_decay=0.2,
    amsgrad=True
)
visenclr_multi=0.1, # clip image backbone lr = lr*visenclr_multi
lanenclr_multi=0.1, # clip text backbone lr = lr*lanenclr_multi
grad_norm_clip = None

scheduler_config = dict(
    type='MultiStepLRWarmUp',
    warmup_epochs=1,
    decay_steps=[35], # this brings from CRIS(CVPR'22)
    decay_ratio=0.1, # this brings from CRIS(CVPR'22)
    max_epoch=50 # this brings from CRIS(CVPR'22)
)