dataset = 'RefCOCOUNC'
data_root = './data/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageAnnotationsFromFile',
         max_token=15, with_mask=True, dataset="RefCOCOUNC"),
    dict(type='LargeScaleJitter', out_max_size=640,
         jitter_min=0.3, jitter_max=1.4),
    # dict(type='Resize', img_scale=(640, 640)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='CollectData', keys=[
         'img', 'ref_expr_inds','gt_mask'])
]
val_pipeline = [
    dict(type='LoadImageAnnotationsFromFile',
         max_token=15, with_mask=True, dataset="RefCOCOUNC"),
    dict(type='Resize', img_scale=(640, 640)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='CollectData', keys=[
         'img', 'ref_expr_inds','gt_mask'])
]
test_pipeline = val_pipeline.copy()

word_emb_cfg = dict(type='GloVe')
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset,
        which_set='train',
        img_source=['coco'],
        annsfile=data_root + 'annotations/refcoco-unc/instances.json',
        imgsfile=data_root + 'images/mscoco/train2014',
        pipeline=train_pipeline,
        word_emb_cfg=word_emb_cfg),
    val=dict(
        type=dataset,
        which_set='val',
        img_source=['coco'],
        annsfile=data_root + 'annotations/refcoco-unc/instances.json',
        imgsfile=data_root + 'images/mscoco/train2014',
        pipeline=val_pipeline,
        word_emb_cfg=word_emb_cfg),
    testA=dict(
        type=dataset,
        which_set='testA',
        img_source=['coco'],
        annsfile=data_root + 'annotations/refcoco-unc/instances.json',
        imgsfile=data_root + 'images/mscoco/train2014',
        pipeline=test_pipeline,
        word_emb_cfg=word_emb_cfg),
    testB=dict(
        type=dataset,
        which_set='testB',
        img_source=['coco'],
        annsfile=data_root + 'annotations/refcoco-unc/instances.json',
        imgsfile=data_root + 'images/mscoco/train2014',
        pipeline=test_pipeline,
        word_emb_cfg=word_emb_cfg)
)
