#!/bin/bash

function main() {
  python3 main.py fit --config=config/sota_fmnist.yaml \
    --data.init_args.root_dir="$DATA_PATH/Fashion-MNIST/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=64 \
    --model.init_args.image_size=64 \
    --model.init_args.sota_config_path="./config/huggingface/fmnist/resnet-18.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_fmnist.yaml \
    --data.init_args.root_dir="$DATA_PATH/Fashion-MNIST/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=64 \
    --model.init_args.image_size=64 \
    --model.init_args.sota_config_path="./config/huggingface/fmnist/resnet-50.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_fmnist.yaml \
    --data.init_args.root_dir="$DATA_PATH/Fashion-MNIST/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=64 \
    --model.init_args.image_size=64 \
    --model.init_args.sota_config_path="./config/huggingface/fmnist/efficientnet-b0.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_fmnist.yaml \
    --data.init_args.root_dir="$DATA_PATH/Fashion-MNIST/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=64 \
    --model.init_args.image_size=64 \
    --model.init_args.sota_config_path="./config/huggingface/fmnist/efficientnet-b3.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_fmnist.yaml \
    --data.init_args.root_dir="$DATA_PATH/Fashion-MNIST/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=64 \
    --model.init_args.image_size=64 \
    --model.init_args.sota_config_path="./config/huggingface/fmnist/mobilenetv2-1.0.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_fmnist.yaml \
    --data.init_args.root_dir="$DATA_PATH/Fashion-MNIST/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=64 \
    --model.init_args.image_size=64 \
    --model.init_args.sota_config_path="./config/huggingface/fmnist/mobilenetv2-1.4.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_fmnist.yaml \
    --data.init_args.root_dir="$DATA_PATH/Fashion-MNIST/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=32 \
    --model.init_args.image_size=32 \
    --model.init_args.optimizer_args.lr_base=0.0001 \
    --model.init_args.sota_config_path="./config/huggingface/fmnist/vit-base.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_fmnist.yaml \
    --data.init_args.root_dir="$DATA_PATH/Fashion-MNIST/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=64 \
    --model.init_args.image_size=64 \
    --model.init_args.sota_config_path="./config/huggingface/fmnist/mobilevit-xs.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_fmnist.yaml \
    --data.init_args.root_dir="$DATA_PATH/Fashion-MNIST/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=64 \
    --model.init_args.image_size=64 \
    --model.init_args.sota_config_path="./config/huggingface/fmnist/mobilevit-s.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_fmnist.yaml \
    --data.init_args.root_dir="$DATA_PATH/Fashion-MNIST/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=64 \
    --model.init_args.image_size=64 \
    --model.init_args.sota_config_path="./config/huggingface/fmnist/efficientformer-l1.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_fmnist.yaml \
    --data.init_args.root_dir="$DATA_PATH/Fashion-MNIST/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=64 \
    --model.init_args.image_size=64 \
    --model.init_args.sota_config_path="./config/huggingface/fmnist/efficientformer-l3.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_fmnist.yaml \
    --data.init_args.root_dir="$DATA_PATH/Fashion-MNIST/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=64 \
    --model.init_args.image_size=64 \
    --model.init_args.sota_config_path="./config/huggingface/fmnist/swiftformer-xs.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_fmnist.yaml \
    --data.init_args.root_dir="$DATA_PATH/Fashion-MNIST/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=64 \
    --model.init_args.image_size=64 \
    --model.init_args.sota_config_path="./config/huggingface/fmnist/swiftformer-s.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_fmnist.yaml \
    --data.init_args.root_dir="$DATA_PATH/Fashion-MNIST/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=32 \
    --model.init_args.image_size=32 \
    --model.init_args.sota_config_path="./config/huggingface/fmnist/cvt-13.json" \
    --trainer.enable_progress_bar=False

  python3 main.py fit --config=config/sota_fmnist.yaml \
    --data.init_args.root_dir="$DATA_PATH/Fashion-MNIST/" \
    --data.init_args.batch_size=512 \
    --data.init_args.image_size=32 \
    --model.init_args.image_size=32 \
    --model.init_args.sota_config_path="./config/huggingface/fmnist/cvt-21.json" \
    --trainer.enable_progress_bar=False
}

rm -f ./script.log
main > script.log 2>&1 &

