CUDA_VISIBLE_DEVICES=0,1 python main_mixup.py --penalty_weight 0.2 --irm_weight_maxim 0.5 --maximize_iter 50 --random_init --constrain --constrain_relax --dataset STL --epochs 400 --keep_cont --offline --retain_group --name IPIRM_STL_mixup_400epoch --ours_mixup_mode full --temperature 0.2
CUDA_VISIBLE_DEVICES=0,1 python linear.py --model_path results/STL/IPIRM_STL_mixup_400epoch/model_400.pth --dataset STL --txt --name IPIRM_STL_mixup_400epoch

