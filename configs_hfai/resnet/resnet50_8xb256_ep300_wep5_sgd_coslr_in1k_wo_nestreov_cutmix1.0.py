_base_ = [
    './models/resnet50_soft.py',
    './datasets/imagenet_bs64.py',
    './schedules/imagenet_bs512_sgd_coslr_ep300.py',
    './runtimes/default_runtime.py'
]


data = dict(
    samples_per_gpu=256,
    workers_per_gpu=4
)

# model settings
model = dict(
    train_cfg=dict(
        augments=dict(type='BatchCutMix', alpha=1.0, num_classes=1000, prob=0.5)))


optimizer = dict(
    type='SGD', lr=0.8)

lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=5*626,
    warmup_ratio=0.25
    )