CUDA_VISIBLE_DEVICES=0,1 python main_mixup.py --baseline --dataset STL --epochs 400 --name Baseline_STL_mixup_epoch400
CUDA_VISIBLE_DEVICES=0,1 python linear.py --model_path results/STL/Baseline_STL_mixup_epoch400/model_400.pth --dataset STL --txt --name Baseline_STL_mixup_epoch400
