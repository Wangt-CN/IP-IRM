CUDA_VISIBLE_DEVICES=0,1 python main.py --penalty_weight 0.2 --irm_weight_maxim 0.5 --maximize_iter 50 --random_init --constrain --constrain_relax --dataset STL --epochs 400 --offline --keep_cont --retain_group --name IPIRM_STL_epoch400
CUDA_VISIBLE_DEVICES=0,1 python linear.py --model_path results/STL/IPIRM_STL_epoch400/model_400.pth --dataset STL --txt --name IPIRM_STL_epoch400
