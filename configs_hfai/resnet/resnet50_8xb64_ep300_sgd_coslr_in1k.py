_base_ = [
    './models/resnet50.py',
    './datasets/imagenet_bs64.py',
    './schedules/imagenet_bs512_sgd_coslr_ep300.py',
    './runtimes/default_runtime.py'
]
