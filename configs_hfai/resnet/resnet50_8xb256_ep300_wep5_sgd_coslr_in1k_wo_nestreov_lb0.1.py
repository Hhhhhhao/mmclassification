_base_ = [
    './models/resnet50.py',
    './datasets/imagenet_bs64.py',
    './schedules/imagenet_bs512_sgd_coslr_ep300.py',
    './runtimes/default_runtime.py'
]


data = dict(
    samples_per_gpu=256,
    workers_per_gpu=4
)

model = dict(
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=2048,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, loss_weight=1.0),
        topk=(1, 5),
    ))


optimizer = dict(
    type='SGD', lr=0.8)

lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=5*626,
    warmup_ratio=0.25
    )