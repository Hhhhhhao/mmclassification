_base_ = [
    './models/resnet50.py',
    './datasets/imagenet_bs64.py',
    './schedules/imagenet_bs2048_lamb_coslr_ep300.py',
    './runtimes/default_runtime.py'
]


data = dict(
    samples_per_gpu=256,
    workers_per_gpu=4
)


optimizer = dict(type='Lamb', lr=0.005, weight_decay=1e-3)
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1.0e-6,
    warmup='linear',
    # For ImageNet-1k, 626 iters per epoch, warmup 5 epochs.
    warmup_iters=5 * 626,
    warmup_ratio=0.0001)