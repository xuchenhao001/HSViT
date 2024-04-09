#!/bin/bash

function per_attn_64_conv_kernel() {
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[512]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.conv_strides="[2]" --model.init_args.attn_num=8 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
}

function per_attn_128_conv_kernel() {
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[512]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.conv_strides="[2]" --model.init_args.attn_num=4 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[1024]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.conv_strides="[2]" --model.init_args.attn_num=8 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
}

function per_attn_256_conv_kernel() {
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[512]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.conv_strides="[2]" --model.init_args.attn_num=2 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[1024]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.conv_strides="[2]" --model.init_args.attn_num=4 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[2048]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.conv_strides="[2]" --model.init_args.attn_num=8 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
}

rm -f ./script.log
per_attn_64_conv_kernel > script.log
per_attn_128_conv_kernel > script.log
per_attn_256_conv_kernel > script.log
