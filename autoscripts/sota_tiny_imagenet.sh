#!/bin/bash

function main() {
  python3 main.py fit --config=config/sota_tiny_imagenet.yaml \
    --data.init_args.root_dir="$DATA_PATH/Tiny-ImageNet/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=64 \
    --model.init_args.image_size=64 \
    --model.init_args.sota_config_path="./config/huggingface/tiny-imagenet/resnet-18.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_tiny_imagenet.yaml \
    --data.init_args.root_dir="$DATA_PATH/Tiny-ImageNet/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=64 \
    --model.init_args.image_size=64 \
    --model.init_args.sota_config_path="./config/huggingface/tiny-imagenet/resnet-50.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_tiny_imagenet.yaml \
    --data.init_args.root_dir="$DATA_PATH/Tiny-ImageNet/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=64 \
    --model.init_args.image_size=64 \
    --model.init_args.sota_config_path="./config/huggingface/tiny-imagenet/efficientnet-b0.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_tiny_imagenet.yaml \
    --data.init_args.root_dir="$DATA_PATH/Tiny-ImageNet/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=64 \
    --model.init_args.image_size=64 \
    --model.init_args.sota_config_path="./config/huggingface/tiny-imagenet/efficientnet-b3.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_tiny_imagenet.yaml \
    --data.init_args.root_dir="$DATA_PATH/Tiny-ImageNet/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=64 \
    --model.init_args.image_size=64 \
    --model.init_args.sota_config_path="./config/huggingface/tiny-imagenet/mobilenetv2-1.0.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_tiny_imagenet.yaml \
    --data.init_args.root_dir="$DATA_PATH/Tiny-ImageNet/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=64 \
    --model.init_args.image_size=64 \
    --model.init_args.sota_config_path="./config/huggingface/tiny-imagenet/mobilenetv2-1.4.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_tiny_imagenet.yaml \
    --data.init_args.root_dir="$DATA_PATH/Tiny-ImageNet/" \
    --data.init_args.batch_size=128 \
    --data.init_args.image_size=64 \
    --model.init_args.image_size=64 \
    --model.init_args.optimizer_args.lr_base=0.0001 \
    --model.init_args.sota_config_path="./config/huggingface/tiny-imagenet/vit-base.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_tiny_imagenet.yaml \
    --data.init_args.root_dir="$DATA_PATH/Tiny-ImageNet/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=64 \
    --model.init_args.image_size=64 \
    --model.init_args.sota_config_path="./config/huggingface/tiny-imagenet/mobilevit-xs.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_tiny_imagenet.yaml \
    --data.init_args.root_dir="$DATA_PATH/Tiny-ImageNet/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=64 \
    --model.init_args.image_size=64 \
    --model.init_args.sota_config_path="./config/huggingface/tiny-imagenet/mobilevit-s.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_tiny_imagenet.yaml \
    --data.init_args.root_dir="$DATA_PATH/Tiny-ImageNet/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=64 \
    --model.init_args.image_size=64 \
    --model.init_args.sota_config_path="./config/huggingface/tiny-imagenet/efficientformer-l1.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_tiny_imagenet.yaml \
    --data.init_args.root_dir="$DATA_PATH/Tiny-ImageNet/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=64 \
    --model.init_args.image_size=64 \
    --model.init_args.sota_config_path="./config/huggingface/tiny-imagenet/efficientformer-l3.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_tiny_imagenet.yaml \
    --data.init_args.root_dir="$DATA_PATH/Tiny-ImageNet/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=64 \
    --model.init_args.image_size=64 \
    --model.init_args.sota_config_path="./config/huggingface/tiny-imagenet/swiftformer-xs.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_tiny_imagenet.yaml \
    --data.init_args.root_dir="$DATA_PATH/Tiny-ImageNet/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=64 \
    --model.init_args.image_size=64 \
    --model.init_args.sota_config_path="./config/huggingface/tiny-imagenet/swiftformer-s.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_tiny_imagenet.yaml \
    --data.init_args.root_dir="$DATA_PATH/Tiny-ImageNet/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=64 \
    --model.init_args.image_size=64 \
    --model.init_args.sota_config_path="./config/huggingface/tiny-imagenet/cvt-13.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_tiny_imagenet.yaml \
    --data.init_args.root_dir="$DATA_PATH/Tiny-ImageNet/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=64 \
    --model.init_args.image_size=64 \
    --model.init_args.sota_config_path="./config/huggingface/tiny-imagenet/cvt-21.json" \
    --trainer.enable_progress_bar=False
}

rm -f ./script.log
main > script.log 2>&1 &

