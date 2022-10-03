_base_ = [
    './models/resnet50.py',
    './datasets/imagenet_bs256_rsb_a12.py',
    './schedules/imagenet_bs2048_lamb_coslr.py',
    './runtimes/default_runtime.py'
]


# Model settings
model = dict(
    backbone=dict(
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        drop_path_rate=0.05,
    ),
    head=dict(loss=dict(use_sigmoid=True)),
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.1, num_classes=1000, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=1000, prob=0.5)
    ]))

# Dataset settings
sampler = dict(type='RepeatAugSampler')

# Schedule settings
runner = dict(max_epochs=300)
optimizer = dict(paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.))
