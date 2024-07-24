# source ./switch-cuda/switch-cuda.sh 11.6
# conda activate FSGS
# pip install tqdm plyfile==0.8.1
# pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
# cp -a /home/laurence/SparseGS_aaai25/submodules/diff-gaussian-rasterization-feat ./submodules/
# pip install Pillow==9.5.0


# 3dgs
# --depth_weight=0.0 --args.depth_pseudo_weight=0.0






# Ours + Feature_loss
export SCENE=horns; EXP_NAME=${SCENE}_fsgs_ours1; 
CUDA_VISIBLE_DEVICES=2 python train.py  --source_path dataset/nerf_llff_data/$SCENE --model_path output/$EXP_NAME --eval  --n_views 3 --sample_pseudo_interval 1 --load_vgg_img1k_model=1 --use_feat_l1_loss=1 --lambda_feat_l1_loss=1.0 --load_vgg_img1k_upto=1 --feat_map_wh_scale=2
# _fsgs: "SSIM": 0.7178561687469482, "PSNR": 20.426536560058594, "LPIPS": 0.22451944090425968
# ------------------------------------------------------------
#   "SSIM": 0.6799218654632568, "PSNR": 19.248538970947266, "LPIPS": 0.2907624188810587
# (ours2dT2dFQB2): --lambda_feat_l1_loss=0.05 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat)' --test_iterations=10_000 --save_iterations=10_000 --end_feat_reg=9500
#   "SSIM": 0.7124350070953369, "PSNR": 20.113515853881836, "LPIPS": 0.22594724409282207
# (ours2cT2dFQB2): --lambda_feat_l1_loss=0.1 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat)' --test_iterations=10_000 --save_iterations=10_000 --end_feat_reg=9500
#   "SSIM": 0.7189317345619202, "PSNR": 20.398094177246094, "LPIPS": 0.2237066738307476
# (ours2gT2dFQB2): --lambda_feat_l1_loss=0.005 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat)' --test_iterations=10_000 --save_iterations=10_000 --end_feat_reg=9500
#   "SSIM": 0.7127436995506287, "PSNR": 20.39345932006836, "LPIPS": 0.2295185849070549
# (ours2aT2dFQB2 👍): --lambda_feat_l1_loss=0.5 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat)' --test_iterations=10_000 --save_iterations=10_000 --end_feat_reg=9500 
#   "SSIM": 0.7242465019226074, "PSNR": 20.41090202331543, "LPIPS": 0.22182752192020416
# (ours2T2FQB2): --lambda_feat_l1_loss=1.0 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat)' --test_iterations=10_000 --save_iterations=10_000 --end_feat_reg=9500
#   "SSIM": 0.7144064903259277, "PSNR": 20.442977905273438, "LPIPS": 0.2290863674134016
# (ours2kT2FQB2): --lambda_feat_l1_loss=2.0 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat)' --test_iterations=10_000 --save_iterations=10_000 --end_feat_reg=9500
#   "SSIM": 0.711108386516571, "PSNR": 20.345441818237305, "LPIPS": 0.23123489692807198
# ------------------------------------------------------------
# (ours2aT2U3dFQB2): --lambda_feat_l1_loss=0.5 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat)' --test_iterations=10_000 --save_iterations=10_000 --end_feat_reg=9500 --load_vgg_img1k_upto=3
#   "SSIM": 0.7222380042076111, "PSNR": 20.42491340637207, "LPIPS": 0.22520375065505505
# (ours2aT2U6dFQB2): --lambda_feat_l1_loss=0.5 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat)' --test_iterations=10_000 --save_iterations=10_000 --end_feat_reg=9500 --load_vgg_img1k_upto=6 
#   "SSIM": 0.7114356160163879, "PSNR": 20.171205520629883, "LPIPS": 0.23516614362597466
# (ours2aT2U8dFQB2): --lambda_feat_l1_loss=0.5 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat)' --test_iterations=10_000 --save_iterations=10_000 --end_feat_reg=9500 --load_vgg_img1k_upto=8 
# ------------------------------------------------------------
# (ours2aT2QB2): --lambda_feat_l1_loss=0.5 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59' --test_iterations=10_000 --save_iterations=10_000 --end_feat_reg=9500
#   "SSIM": 0.7166590094566345, "PSNR": 20.31754493713379, "LPIPS": 0.2236715368926525
# ------------------------------------------------------------
# ours2aT2U8dFQB:  --lambda_feat_l1_loss=0.5 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat)' --test_iterations=10_000 --save_iterations=10_000 --end_feat_reg=9500 --load_vgg_img1k_upto=8
#   "SSIM": 0.707737922668457, "PSNR": 20.383867263793945, "LPIPS": 0.2370066624134779






# Evaluation
export EXP_NAME=horns_fsgs; SCENE=horns; CUDA_VISIBLE_DEVICES=0
python render.py --source_path dataset/nerf_llff_data/${SCENE}/  --model_path  output/${EXP_NAME} --iteration 10000
python metrics.py --source_path dataset/nerf_llff_data/${EXP_NAME}/  --model_path  output/${EXP_NAME} --iteration 10000




