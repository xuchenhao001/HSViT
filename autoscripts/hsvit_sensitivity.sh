#!/bin/bash

function conv1attn1() {
  #  conv_depth=1, attn_depth=1
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[16]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=0 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[32]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=0 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[64]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=0 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[128]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=0 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[256]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=0 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[16]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=1 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[32]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=1 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[64]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=1 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[128]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=1 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[256]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=1 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[16]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=2 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[32]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=2 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[64]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=2 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[128]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=2 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[256]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=2 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[16]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=4 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[32]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=4 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[64]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=4 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[128]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=4 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[256]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=4 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[16]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=8 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[32]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=8 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[64]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=8 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[128]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=8 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[256]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=8 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
}

function conv1attn2() {
  # conv_depth=1, attn_depth=2
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[16]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=1 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[32]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=1 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[64]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=1 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[128]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=1 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[256]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=1 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[16]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=2 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[32]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=2 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[64]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=2 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[128]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=2 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[256]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=2 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[16]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=4 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[32]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=4 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[64]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=4 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[128]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=4 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[256]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=4 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[16]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=8 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[32]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=8 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[64]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=8 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[128]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=8 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[256]" --model.init_args.conv_kernel_sizes="[3]" --model.init_args.pool_strides="[2]" --model.init_args.attn_num=8 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False

}


function conv2attn1() {
  #  conv_depth=2, attn_depth=1
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[8,16]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=1 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[16,32]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=1 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[32,64]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=1 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[64,128]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=1 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[128,256]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=1 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[8,16]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=2 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[16,32]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=2 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[32,64]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=2 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[64,128]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=2 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[128,256]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=2 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[8,16]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=4 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[16,32]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=4 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[32,64]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=4 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[64,128]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=4 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[128,256]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=4 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[8,16]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=8 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[16,32]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=8 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[32,64]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=8 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[64,128]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=8 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[128,256]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=8 --model.init_args.attn_depth=1 --trainer.enable_progress_bar=False

}

function conv2attn2() {
  #  conv_depth=2, attn_depth=2
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[8,16]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=1 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[16,32]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=1 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[32,64]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=1 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[64,128]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=1 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[128,256]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=1 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[8,16]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=2 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[16,32]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=2 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[32,64]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=2 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[64,128]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=2 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[128,256]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=2 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[8,16]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=4 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[16,32]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=4 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[32,64]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=4 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[64,128]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=4 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[128,256]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=4 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[8,16]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=8 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[16,32]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=8 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[32,64]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=8 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[64,128]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=8 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False
  python3 main.py fit --config=config/hsvit_cifar10.yaml --model.init_args.conv_kernel_nums="[128,256]" --model.init_args.conv_kernel_sizes="[3,3]" --model.init_args.pool_strides="[2,2]" --model.init_args.attn_num=8 --model.init_args.attn_depth=2 --trainer.enable_progress_bar=False

}


rm -f ./script.log
conv2attn2 > script.log
