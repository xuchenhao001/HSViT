#!/bin/bash

function cifar10() {
  python3 main.py fit --config=config/hsvit_cifar10.yaml \
    --model.init_args.conv_kernel_nums="[64, 128]" \
    --model.init_args.conv_kernel_sizes="[3, 3]" \
    --model.init_args.pool_strides="[2, 2]" \
    --model.init_args.attn_num=16 \
    --model.init_args.attn_depth=2 \
    --model.init_args.attn_embed_dim=64 \
    --model.init_args.dropout=0.2 \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/hsvit_cifar10.yaml \
    --model.init_args.conv_kernel_nums="[64, 128, 256]" \
    --model.init_args.conv_kernel_sizes="[3, 3, 3]" \
    --model.init_args.pool_strides="[2, 1, 2]" \
    --model.init_args.attn_num=16 \
    --model.init_args.attn_depth=4 \
    --model.init_args.attn_embed_dim=64 \
    --model.init_args.dropout=0.2 \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/hsvit_cifar10.yaml \
    --model.init_args.conv_kernel_nums="[64, 128, 256, 512]" \
    --model.init_args.conv_kernel_sizes="[3, 3, 3, 3]" \
    --model.init_args.pool_strides="[2, 1, 2, 1]" \
    --model.init_args.attn_num=16 \
    --model.init_args.attn_depth=8 \
    --model.init_args.attn_embed_dim=64 \
    --model.init_args.dropout=0.2 \
    --trainer.enable_progress_bar=False
}

function cifar100() {
  python3 main.py fit --config=config/hsvit_cifar100.yaml \
    --model.init_args.conv_kernel_nums="[64, 128]" \
    --model.init_args.conv_kernel_sizes="[3, 3]" \
    --model.init_args.pool_strides="[2, 2]" \
    --model.init_args.attn_num=16 \
    --model.init_args.attn_depth=2 \
    --model.init_args.attn_embed_dim=64 \
    --model.init_args.dropout=0.2 \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/hsvit_cifar100.yaml \
    --model.init_args.conv_kernel_nums="[64, 128, 256]" \
    --model.init_args.conv_kernel_sizes="[3, 3, 3]" \
    --model.init_args.pool_strides="[2, 1, 2]" \
    --model.init_args.attn_num=16 \
    --model.init_args.attn_depth=4 \
    --model.init_args.attn_embed_dim=64 \
    --model.init_args.dropout=0.2 \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/hsvit_cifar100.yaml \
    --model.init_args.conv_kernel_nums="[64, 128, 256, 512]" \
    --model.init_args.conv_kernel_sizes="[3, 3, 3, 3]" \
    --model.init_args.pool_strides="[2, 1, 2, 1]" \
    --model.init_args.attn_num=16 \
    --model.init_args.attn_depth=8 \
    --model.init_args.attn_embed_dim=64 \
    --model.init_args.dropout=0.2 \
    --trainer.enable_progress_bar=False
}

function fmnist() {
  python3 main.py fit --config=config/hsvit_fmnist.yaml \
    --model.init_args.conv_kernel_nums="[64, 128]" \
    --model.init_args.conv_kernel_sizes="[3, 3]" \
    --model.init_args.pool_strides="[2, 2]" \
    --model.init_args.attn_num=16 \
    --model.init_args.attn_depth=2 \
    --model.init_args.attn_embed_dim=64 \
    --model.init_args.dropout=0.2 \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/hsvit_fmnist.yaml \
    --model.init_args.conv_kernel_nums="[64, 128, 256]" \
    --model.init_args.conv_kernel_sizes="[3, 3, 3]" \
    --model.init_args.pool_strides="[2, 1, 2]" \
    --model.init_args.attn_num=16 \
    --model.init_args.attn_depth=4 \
    --model.init_args.attn_embed_dim=64 \
    --model.init_args.dropout=0.2 \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/hsvit_fmnist.yaml \
    --model.init_args.conv_kernel_nums="[64, 128, 256, 512]" \
    --model.init_args.conv_kernel_sizes="[3, 3, 3, 3]" \
    --model.init_args.pool_strides="[2, 1, 2, 1]" \
    --model.init_args.attn_num=16 \
    --model.init_args.attn_depth=8 \
    --model.init_args.attn_embed_dim=64 \
    --model.init_args.dropout=0.2 \
    --trainer.enable_progress_bar=False
}

function tiny_imagenet() {
  python3 main.py fit --config=config/hsvit_tiny_imagenet.yaml \
    --model.init_args.conv_kernel_nums="[64, 128]" \
    --model.init_args.conv_kernel_sizes="[3, 3]" \
    --model.init_args.pool_strides="[4, 2]" \
    --model.init_args.attn_num=16 \
    --model.init_args.attn_depth=2 \
    --model.init_args.attn_embed_dim=64 \
    --model.init_args.dropout=0.2 \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/hsvit_tiny_imagenet.yaml \
    --model.init_args.conv_kernel_nums="[64, 128, 256]" \
    --model.init_args.conv_kernel_sizes="[3, 3, 3]" \
    --model.init_args.pool_strides="[2, 2, 2]" \
    --model.init_args.attn_num=16 \
    --model.init_args.attn_depth=4 \
    --model.init_args.attn_embed_dim=64 \
    --model.init_args.dropout=0.2 \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/hsvit_tiny_imagenet.yaml \
    --model.init_args.conv_kernel_nums="[64, 128, 256, 512]" \
    --model.init_args.conv_kernel_sizes="[3, 3, 3, 3]" \
    --model.init_args.pool_strides="[2, 2, 2, 1]" \
    --model.init_args.attn_num=16 \
    --model.init_args.attn_depth=8 \
    --model.init_args.attn_embed_dim=64 \
    --model.init_args.dropout=0.2 \
    --trainer.enable_progress_bar=False
}

function food101() {
  python3 main.py fit --config=config/hsvit_food101.yaml \
    --model.init_args.conv_kernel_nums="[64, 128]" \
    --model.init_args.conv_kernel_sizes="[3, 3]" \
    --model.init_args.pool_strides="[4, 4]" \
    --model.init_args.attn_num=16 \
    --model.init_args.attn_depth=2 \
    --model.init_args.attn_embed_dim=64 \
    --model.init_args.dropout=0.2 \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/hsvit_food101.yaml \
    --model.init_args.conv_kernel_nums="[64, 128, 256]" \
    --model.init_args.conv_kernel_sizes="[3, 3, 3]" \
    --model.init_args.pool_strides="[4, 2, 2]" \
    --model.init_args.attn_num=16 \
    --model.init_args.attn_depth=4 \
    --model.init_args.attn_embed_dim=64 \
    --model.init_args.dropout=0.2 \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/hsvit_food101.yaml \
    --model.init_args.conv_kernel_nums="[64, 128, 256, 512]" \
    --model.init_args.conv_kernel_sizes="[3, 3, 3, 3]" \
    --model.init_args.pool_strides="[2, 2, 2, 2]" \
    --model.init_args.attn_num=16 \
    --model.init_args.attn_depth=8 \
    --model.init_args.attn_embed_dim=64 \
    --model.init_args.dropout=0.2 \
    --trainer.enable_progress_bar=False
}

function main() {
  cifar10
  cifar100
  fmnist
  tiny_imagenet
  food101
}

rm -f ./script.log
main > script.log 2>&1 &

