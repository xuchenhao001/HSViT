#!/bin/bash

function main() {
  python3 main.py fit --config=config/sota_cifar100.yaml \
    --data.init_args.root_dir="$DATA_PATH/CIFAR100/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=64 \
    --model.init_args.image_size=64 \
    --model.init_args.sota_config_path="./config/huggingface/cifar100/resnet-18.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_cifar100.yaml \
    --data.init_args.root_dir="$DATA_PATH/CIFAR100/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=64 \
    --model.init_args.image_size=64 \
    --model.init_args.sota_config_path="./config/huggingface/cifar100/resnet-50.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_cifar100.yaml \
    --data.init_args.root_dir="$DATA_PATH/CIFAR100/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=64 \
    --model.init_args.image_size=64 \
    --model.init_args.sota_config_path="./config/huggingface/cifar100/efficientnet-b0.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_cifar100.yaml \
    --data.init_args.root_dir="$DATA_PATH/CIFAR100/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=64 \
    --model.init_args.image_size=64 \
    --model.init_args.sota_config_path="./config/huggingface/cifar100/efficientnet-b3.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_cifar100.yaml \
    --data.init_args.root_dir="$DATA_PATH/CIFAR100/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=64 \
    --model.init_args.image_size=64 \
    --model.init_args.sota_config_path="./config/huggingface/cifar100/mobilenetv2-1.0.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_cifar100.yaml \
    --data.init_args.root_dir="$DATA_PATH/CIFAR100/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=64 \
    --model.init_args.image_size=64 \
    --model.init_args.sota_config_path="./config/huggingface/cifar100/mobilenetv2-1.4.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_cifar100.yaml \
    --data.init_args.root_dir="$DATA_PATH/CIFAR100/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=32 \
    --model.init_args.image_size=32 \
    --model.init_args.optimizer_args.lr_base=0.0001 \
    --model.init_args.sota_config_path="./config/huggingface/cifar100/vit-base.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_cifar100.yaml \
    --data.init_args.root_dir="$DATA_PATH/CIFAR100/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=64 \
    --model.init_args.image_size=64 \
    --model.init_args.sota_config_path="./config/huggingface/cifar100/mobilevit-xs.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_cifar100.yaml \
    --data.init_args.root_dir="$DATA_PATH/CIFAR100/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=64 \
    --model.init_args.image_size=64 \
    --model.init_args.sota_config_path="./config/huggingface/cifar100/mobilevit-s.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_cifar100.yaml \
    --data.init_args.root_dir="$DATA_PATH/CIFAR100/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=64 \
    --model.init_args.image_size=64 \
    --model.init_args.sota_config_path="./config/huggingface/cifar100/efficientformer-l1.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_cifar100.yaml \
    --data.init_args.root_dir="$DATA_PATH/CIFAR100/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=64 \
    --model.init_args.image_size=64 \
    --model.init_args.sota_config_path="./config/huggingface/cifar100/efficientformer-l3.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_cifar100.yaml \
    --data.init_args.root_dir="$DATA_PATH/CIFAR100/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=64 \
    --model.init_args.image_size=64 \
    --model.init_args.sota_config_path="./config/huggingface/cifar100/swiftformer-xs.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_cifar100.yaml \
    --data.init_args.root_dir="$DATA_PATH/CIFAR100/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=64 \
    --model.init_args.image_size=64 \
    --model.init_args.sota_config_path="./config/huggingface/cifar100/swiftformer-s.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_cifar100.yaml \
    --data.init_args.root_dir="$DATA_PATH/CIFAR100/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=32 \
    --model.init_args.image_size=32 \
    --model.init_args.sota_config_path="./config/huggingface/cifar100/cvt-13.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_cifar100.yaml \
    --data.init_args.root_dir="$DATA_PATH/CIFAR100/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=32 \
    --model.init_args.image_size=32 \
    --model.init_args.sota_config_path="./config/huggingface/cifar100/cvt-21.json" \
    --trainer.enable_progress_bar=False
}

rm -f ./script.log
main > script.log 2>&1 &

