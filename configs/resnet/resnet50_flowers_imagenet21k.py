_base_ = [
    '../_base_/models/resnet50_flowers.py', '../_base_/datasets/flowers_bs32.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        frozen_stages=2,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='./resources/resnet50_imagenet21k.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=17),
)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# 学习率衰减策略
lr_config = dict(policy='step', step=[15])
runner = dict(type='EpochBasedRunner', max_epochs=20)
log_config = dict(interval=100)