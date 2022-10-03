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


optimizer = dict(
    type='SGD', lr=0.8, momentum=0.9, weight_decay=0.0001, nesterov=True)


# Model settings
model = dict(
    backbone=dict(
        norm_cfg=dict(type='SyncBN', requires_grad=True),
    ))