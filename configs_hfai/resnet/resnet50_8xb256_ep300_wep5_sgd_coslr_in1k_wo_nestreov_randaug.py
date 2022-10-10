_base_ = [
    './models/resnet50.py',
    './datasets/imagenet_bs64_randaug.py',
    './schedules/imagenet_bs512_sgd_coslr_ep300.py',
    './runtimes/default_runtime.py'
]


data = dict(
    samples_per_gpu=256,
    workers_per_gpu=4
)


optimizer = dict(
    type='SGD', lr=0.8)

lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=5*626,
    warmup_ratio=0.25
    )