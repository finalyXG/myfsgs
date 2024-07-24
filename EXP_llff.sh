# source ./switch-cuda/switch-cuda.sh 11.6
# conda activate FSGS
# pip install tqdm plyfile==0.8.1
# pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
# cp -a /home/laurence/SparseGS_aaai25/submodules/diff-gaussian-rasterization-feat ./submodules/



# 3dgs
# --depth_weight=0.0 --args.depth_pseudo_weight=0.0






# "4×", "504 x 378"  
# Baseline
# copy 1
export SCENE=horns; EXP_NAME=${SCENE}_fsgs; 
CUDA_VISIBLE_DEVICES=1 python train.py  --source_path dataset/nerf_llff_data/$SCENE --model_path output/$EXP_NAME --eval  --n_views 3 --sample_pseudo_interval 1
# "SSIM": 0.7178561687469482, "PSNR": 20.426536560058594, "LPIPS": 0.22451944090425968
# (cp1) "SSIM": 0.7169114351272583, "PSNR": 20.556421279907227, "LPIPS": 0.22810371778905392

export SCENE=fern; EXP_NAME=${SCENE}_fsgs; 
CUDA_VISIBLE_DEVICES=2 python train.py  --source_path dataset/nerf_llff_data/$SCENE --model_path output/$EXP_NAME --eval  --n_views 3 --sample_pseudo_interval 1
#   "SSIM": 0.7218281626701355, "PSNR": 21.904983520507812, "LPIPS": 0.20305772125720978
# (cp1):   "SSIM": 0.7248429656028748, "PSNR": 22.011980056762695, "LPIPS": 0.20360970497131348

export SCENE=flower; EXP_NAME=${SCENE}_fsgs; 
CUDA_VISIBLE_DEVICES=0 python train.py  --source_path dataset/nerf_llff_data/$SCENE --model_path output/$EXP_NAME --eval  --n_views 3 --sample_pseudo_interval 1
#   "SSIM": 0.6284586787223816, "PSNR": 20.510786056518555, "LPIPS": 0.24565891176462173
# (cp1):   "SSIM": 0.6308955550193787, "PSNR": 20.429290771484375, "LPIPS": 0.24548400938510895 

export SCENE=fortress; EXP_NAME=${SCENE}_fsgs; 
CUDA_VISIBLE_DEVICES=0 python train.py  --source_path dataset/nerf_llff_data/$SCENE --model_path output/$EXP_NAME --eval  --n_views 3 --sample_pseudo_interval 1
#   "SSIM": 0.724816083908081, "PSNR": 23.419816970825195, "LPIPS": 0.1742019628485044
# (cp1):   "SSIM": 0.7292248606681824, "PSNR": 23.48796844482422, "LPIPS": 0.17087563623984656 

export SCENE=leaves; EXP_NAME=${SCENE}_fsgs; 
CUDA_VISIBLE_DEVICES=0 python train.py  --source_path dataset/nerf_llff_data/$SCENE --model_path output/$EXP_NAME --eval  --n_views 3 --sample_pseudo_interval 1
#   "SSIM": 0.6354213953018188, "PSNR": 17.65214729309082, "LPIPS": 0.2055678814649582
# (cp1):   "SSIM": 0.6144864559173584, "PSNR": 17.115550994873047, "LPIPS": 0.22785510122776031 

export SCENE=orchids; EXP_NAME=${SCENE}_fsgs; 
CUDA_VISIBLE_DEVICES=0 python train.py  --source_path dataset/nerf_llff_data/$SCENE --model_path output/$EXP_NAME --eval  --n_views 3 --sample_pseudo_interval 1
#   "SSIM": 0.5194613933563232, "PSNR": 16.536609649658203, "LPIPS": 0.2571282237768173
# (cp1):   "SSIM": 0.5231111645698547, "PSNR": 16.556520462036133, "LPIPS": 0.25224461406469345 


export SCENE=room; EXP_NAME=${SCENE}_fsgs; 
CUDA_VISIBLE_DEVICES=0 python train.py  --source_path dataset/nerf_llff_data/$SCENE --model_path output/$EXP_NAME --eval  --n_views 3 --sample_pseudo_interval 1
#   "SSIM": 0.8472223281860352, "PSNR": 21.69502830505371, "LPIPS": 0.1668744459748268
# (cp1):   "SSIM": 0.843590259552002, "PSNR": 21.420745849609375, "LPIPS": 0.16556512812773386 

export SCENE=trex; EXP_NAME=${SCENE}_fsgs; 
CUDA_VISIBLE_DEVICES=0 python train.py  --source_path dataset/nerf_llff_data/$SCENE --model_path output/$EXP_NAME --eval  --n_views 3 --sample_pseudo_interval 1
#   "SSIM": 0.7984962463378906, "PSNR": 21.35930061340332, "LPIPS": 0.16756346289600646
# (cp1):   "SSIM": 0.8100265860557556, "PSNR": 21.825254440307617, "LPIPS": 0.15899154863187245 

# % FSGS: avg PSNR: 20.44; SSIM: 0.699; LPIPS: 0.206

# copy 1
export SCENE=horns; EXP_NAME=${SCENE}_fsgs_cp1; 
CUDA_VISIBLE_DEVICES=2 python train.py  --source_path dataset/nerf_llff_data/$SCENE --model_path output/$EXP_NAME --eval  --n_views 3 --sample_pseudo_interval 1

export SCENE=fern; EXP_NAME=${SCENE}_fsgs_cp1; 
CUDA_VISIBLE_DEVICES=2 python train.py  --source_path dataset/nerf_llff_data/$SCENE --model_path output/$EXP_NAME --eval  --n_views 3 --sample_pseudo_interval 1

export SCENE=flower; EXP_NAME=${SCENE}_fsgs_cp1; 
CUDA_VISIBLE_DEVICES=2 python train.py  --source_path dataset/nerf_llff_data/$SCENE --model_path output/$EXP_NAME --eval  --n_views 3 --sample_pseudo_interval 1

export SCENE=fortress; EXP_NAME=${SCENE}_fsgs_cp1; 
CUDA_VISIBLE_DEVICES=2 python train.py  --source_path dataset/nerf_llff_data/$SCENE --model_path output/$EXP_NAME --eval  --n_views 3 --sample_pseudo_interval 1

export SCENE=leaves; EXP_NAME=${SCENE}_fsgs_cp1; 
CUDA_VISIBLE_DEVICES=2 python train.py  --source_path dataset/nerf_llff_data/$SCENE --model_path output/$EXP_NAME --eval  --n_views 3 --sample_pseudo_interval 1

export SCENE=orchids; EXP_NAME=${SCENE}_fsgs_cp1; 
CUDA_VISIBLE_DEVICES=2 python train.py  --source_path dataset/nerf_llff_data/$SCENE --model_path output/$EXP_NAME --eval  --n_views 3 --sample_pseudo_interval 1

export SCENE=room; EXP_NAME=${SCENE}_fsgs_cp1; 
CUDA_VISIBLE_DEVICES=2 python train.py  --source_path dataset/nerf_llff_data/$SCENE --model_path output/$EXP_NAME --eval  --n_views 3 --sample_pseudo_interval 1

export SCENE=trex; EXP_NAME=${SCENE}_fsgs_cp1; 
CUDA_VISIBLE_DEVICES=2 python train.py  --source_path dataset/nerf_llff_data/$SCENE --model_path output/$EXP_NAME --eval  --n_views 3 --sample_pseudo_interval 1


# Ours + Feature_loss
export SCENE=horns; EXP_NAME=${SCENE}_fsgs_ours1; 
CUDA_VISIBLE_DEVICES=2 python train.py  --source_path dataset/nerf_llff_data/$SCENE --model_path output/$EXP_NAME --eval  --n_views 3 --sample_pseudo_interval 1 --load_vgg_img1k_model=1 --use_feat_l1_loss=1 --lambda_feat_l1_loss=1.0 --load_vgg_img1k_upto=1 --feat_map_wh_scale=2
#   "SSIM": 0.6799218654632568, "PSNR": 19.248538970947266, "LPIPS": 0.2907624188810587
# (ours1a): --lambda_feat_l1_loss=0.5 
# "SSIM": 0.6900491714477539, "PSNR": 19.407991409301758, "LPIPS": 0.27257087267935276
# (ours1b): --lambda_feat_l1_loss=0.25 
# "SSIM": 0.6900491714477539, "PSNR": 19.407991409301758, "LPIPS": 0.27257087267935276
# (ours1c): --lambda_feat_l1_loss=0.1 
# "SSIM": 0.7054487466812134, "PSNR": 19.693492889404297, "LPIPS": 0.25085534155368805
# (ours2c): --lambda_feat_l1_loss=0.1 --feat_map_wh_scale=1 
# "SSIM": 0.7070963382720947, "PSNR": 19.631244659423828, "LPIPS": 0.2485731728374958
# (ours2cT2): --lambda_feat_l1_loss=0.1 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear 
# "SSIM": 0.7095291614532471, "PSNR": 20.092546463012695, "LPIPS": 0.24142912775278091
# (ours1d): --lambda_feat_l1_loss=0.01 
# "SSIM": 0.7125908136367798, "PSNR": 19.63851547241211, "LPIPS": 0.24059420078992844
# (ours2d): --lambda_feat_l1_loss=0.01 --feat_map_wh_scale=1
# "SSIM": 0.7167648077011108, "PSNR": 20.2413330078125, "LPIPS": 0.23233133181929588
# (ours2dT2): --lambda_feat_l1_loss=0.01 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear
# "SSIM": 0.7153522968292236, "PSNR": 20.328384399414062, "LPIPS": 0.23654549568891525
# (ours1dT2): --lambda_feat_l1_loss=0.01 
# (ours1e)🚩: --lambda_feat_l1_loss=0.0 
# "SSIM": 0.7118057012557983, "PSNR": 20.314847946166992, "LPIPS": 0.22818952426314354
# (ours2f): --lambda_feat_l1_loss=0.002 --feat_map_wh_scale=1
#   "SSIM": 0.7155267000198364, "PSNR": 20.01047134399414, "LPIPS": 0.23829814791679382
# (ours2g): --lambda_feat_l1_loss=0.005 --feat_map_wh_scale=1
#   "SSIM": 0.718313992023468, "PSNR": 20.043350219726562, "LPIPS": 0.23789683170616627
# (ours2gT2): --lambda_feat_l1_loss=0.005 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear
# (ours2gT2U2): --lambda_feat_l1_loss=0.005 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --load_vgg_img1k_upto=2
# ours2gT2U2[fix]: use relu on the gt feature (still wrong needs to rerun)
#   "SSIM": 0.7163535952568054, "PSNR": 20.349925994873047, "LPIPS": 0.23825465328991413
# (ours2gT2U4): --lambda_feat_l1_loss=0.005 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --load_vgg_img1k_upto=4
#   "SSIM": 0.7108922600746155, "PSNR": 19.988004684448242, "LPIPS": 0.24254820123314857
# (ours2h): --lambda_feat_l1_loss=0.0002 --feat_map_wh_scale=1
# (ours2hT2): --lambda_feat_l1_loss=0.0002 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear
#   "SSIM": 0.7241716384887695, "PSNR": 20.705272674560547, "LPIPS": 0.23100201785564423
# (ours2hT2xF): --lambda_feat_l1_loss=0.0002 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode=59-feat
#   "SSIM": 0.7134584784507751, "PSNR": 20.422096252441406, "LPIPS": 0.22820616513490677
# (ours2hT2dF)👍 where 'dF' means detach 16-dim fetures: --lambda_feat_l1_loss=0.0002 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat)'
#   "SSIM": 0.7220652103424072, "PSNR": 20.65489387512207, "LPIPS": 0.22555612213909626
# (ours2iT2dF) where 'dF' means detach 16-dim fetures: --lambda_feat_l1_loss=0.0005 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat)'
#   "SSIM": 0.719347357749939, "PSNR": 20.54144859313965, "LPIPS": 0.2263430617749691
# (ours2jT2dF) where 'dF' means detach 16-dim fetures: --lambda_feat_l1_loss=0.0001 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat)'
#   "SSIM": 0.7126091122627258, "PSNR": 20.16320037841797, "LPIPS": 0.22825180552899837
# (ours2hT2doF)🧩 where 'doF' means detach the opacity feature and the 16-dim fetures: --lambda_feat_l1_loss=0.0002 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat+o)'
#   "SSIM": 0.7104474306106567, "PSNR": 20.451187133789062, "LPIPS": 0.22923866845667362
# ---------------------------------------------------------------------------------------------------
# (ours2hT2dFQ)👍 where 'Q' means 'quick': --lambda_feat_l1_loss=0.0002 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat)' --test_iterations=10_000 --save_iterations=10_000
#   "SSIM": 0.7165917754173279, "PSNR": 20.494518280029297, "LPIPS": 0.2258994672447443
# (ours2dT2dFQB) where 'B' means 'our own settings will only be used *Before* some iteration steps': --lambda_feat_l1_loss=0.01 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat)' --test_iterations=10_000 --save_iterations=10_000 --end_feat_reg=9500
#   "SSIM": 0.7204510569572449, "PSNR": 20.488357543945312, "LPIPS": 0.22742167487740517
# (ours2dT2dFQB1): --lambda_feat_l1_loss=0.1 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat)' --test_iterations=10_000 --save_iterations=10_000 --end_feat_reg=9500
#   "SSIM": 0.7154334187507629, "PSNR": 20.292709350585938, "LPIPS": 0.22794416919350624
# (ours2dT2dFQB2)👍: --lambda_feat_l1_loss=0.05 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat)' --test_iterations=10_000 --save_iterations=10_000 --end_feat_reg=9500
#   "SSIM": 0.7260749340057373, "PSNR": 20.69753074645996, "LPIPS": 0.22275309264659882
# (ours2dT2dFQB3): --lambda_feat_l1_loss=0.005 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat)' --test_iterations=10_000 --save_iterations=10_000 --end_feat_reg=9500
#   "SSIM": 0.7199643850326538, "PSNR": 20.41196632385254, "LPIPS": 0.22178642638027668




export SCENE=fern; EXP_NAME=${SCENE}_fsgs_ours1; 
CUDA_VISIBLE_DEVICES=2 python train.py  --source_path dataset/nerf_llff_data/$SCENE --model_path output/$EXP_NAME --eval  --n_views 3 --sample_pseudo_interval 1 --load_vgg_img1k_model=1 --use_feat_l1_loss=1 --lambda_feat_l1_loss=1.0 --load_vgg_img1k_upto=1 --feat_map_wh_scale=2
# tobeat🚩:  "SSIM": 0.7218281626701355, "PSNR": 21.904983520507812, "LPIPS": 0.20305772125720978
# (ours2hT2dF)🧩 where 'dF' means detach 16-dim fetures: --lambda_feat_l1_loss=0.0002 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat)'
#   "SSIM": 0.726726233959198, "PSNR": 22.02137565612793, "LPIPS": 0.19819495578606924
# (ours2dT2dFQB2)🧩: --lambda_feat_l1_loss=0.05 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat)' --test_iterations=10_000 --save_iterations=10_000 --end_feat_reg=9500
#   "SSIM": 0.7201073169708252, "PSNR": 21.889755249023438, "LPIPS": 0.20191877583662668
# (ours2dT2U4dFQB3): --lambda_feat_l1_loss=0.005 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat)' --test_iterations=10_000 --save_iterations=10_000 --end_feat_reg=9500 --load_vgg_img1k_upto=4
#   "SSIM": 0.7216873168945312, "PSNR": 21.80047035217285, "LPIPS": 0.20517627894878387
# (ours2dT2U4dFQB4)👍: --lambda_feat_l1_loss=0.005 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat)' --test_iterations=10_000 --save_iterations=10_000 --end_feat_reg=9500 --load_vgg_img1k_upto=7
#   "SSIM": 0.7251437306404114, "PSNR": 21.833066940307617, "LPIPS": 0.20375148952007294
# (ours2dT2U4QB4)👍: --lambda_feat_l1_loss=0.005 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59' --test_iterations=10_000 --save_iterations=10_000 --end_feat_reg=9500 --load_vgg_img1k_upto=7
#   "SSIM": 0.7184858918190002, "PSNR": 21.515609741210938, "LPIPS": 0.21734899779160818 
# (ours2dT2U4dFQB5)👍: --lambda_feat_l1_loss=0.005 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat)' --test_iterations=10_000 --save_iterations=10_000 --end_feat_reg=9500 --load_vgg_img1k_upto=11 --transformation_layer_dim=256


export SCENE=flower; EXP_NAME=${SCENE}_fsgs_ours1; 
CUDA_VISIBLE_DEVICES=2 python train.py  --source_path dataset/nerf_llff_data/$SCENE --model_path output/$EXP_NAME --eval  --n_views 3 --sample_pseudo_interval 1 --load_vgg_img1k_model=1 --use_feat_l1_loss=1 --lambda_feat_l1_loss=1.0 --load_vgg_img1k_upto=1 --feat_map_wh_scale=2
# tobeat🚩:  "SSIM": 0.6284586787223816, "PSNR": 20.510786056518555, "LPIPS": 0.24565891176462173
# (ours2hT2dF)🧩where 'dF' means detach 16-dim fetures: --lambda_feat_l1_loss=0.0002 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat)'
#   "SSIM": 0.6294440627098083, "PSNR": 20.46207046508789, "LPIPS": 0.24259777516126632



export SCENE=fortress; EXP_NAME=${SCENE}_fsgs_ours1; 
CUDA_VISIBLE_DEVICES=2 python train.py  --source_path dataset/nerf_llff_data/$SCENE --model_path output/$EXP_NAME --eval  --n_views 3 --sample_pseudo_interval 1 --load_vgg_img1k_model=1 --use_feat_l1_loss=1 --lambda_feat_l1_loss=1.0 --load_vgg_img1k_upto=1 --feat_map_wh_scale=2
# tobeat🚩:  "SSIM": 0.724816083908081, "PSNR": 23.419816970825195, "LPIPS": 0.1742019628485044
# (ours2hT2dF)🧩 where 'dF' means detach 16-dim fetures: --lambda_feat_l1_loss=0.0002 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat)'
#   "SSIM": 0.7426056265830994, "PSNR": 24.00777244567871, "LPIPS": 0.16802440583705902


export SCENE=leaves; EXP_NAME=${SCENE}_fsgs_ours1; 
CUDA_VISIBLE_DEVICES=2 python train.py  --source_path dataset/nerf_llff_data/$SCENE --model_path output/$EXP_NAME --eval  --n_views 3 --sample_pseudo_interval 1 --load_vgg_img1k_model=1 --use_feat_l1_loss=1 --lambda_feat_l1_loss=1.0 --load_vgg_img1k_upto=1 --feat_map_wh_scale=2
# tobeat🚩:   "SSIM": 0.6354213953018188, "PSNR": 17.65214729309082, "LPIPS": 0.2055678814649582
# (ours2hT2dF)🧩 where 'dF' means detach 16-dim fetures: --lambda_feat_l1_loss=0.0002 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat)'
#   "SSIM": 0.634867787361145, "PSNR": 17.70178985595703, "LPIPS": 0.2056180238723755


export SCENE=orchids; EXP_NAME=${SCENE}_fsgs_ours1; 
CUDA_VISIBLE_DEVICES=2 python train.py  --source_path dataset/nerf_llff_data/$SCENE --model_path output/$EXP_NAME --eval  --n_views 3 --sample_pseudo_interval 1 --load_vgg_img1k_model=1 --use_feat_l1_loss=1 --lambda_feat_l1_loss=1.0 --load_vgg_img1k_upto=1 --feat_map_wh_scale=2
# tobeat🚩:   "SSIM": 0.5194613933563232, "PSNR": 16.536609649658203, "LPIPS": 0.2571282237768173
# (ours2hT2dF)🧩 where 'dF' means detach 16-dim fetures: --lambda_feat_l1_loss=0.0002 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat)'
#   "SSIM": 0.515891969203949, "PSNR": 16.479976654052734, "LPIPS": 0.25780561193823814

export SCENE=room; EXP_NAME=${SCENE}_fsgs_ours1; 
CUDA_VISIBLE_DEVICES=2 python train.py  --source_path dataset/nerf_llff_data/$SCENE --model_path output/$EXP_NAME --eval  --n_views 3 --sample_pseudo_interval 1 --load_vgg_img1k_model=1 --use_feat_l1_loss=1 --lambda_feat_l1_loss=1.0 --load_vgg_img1k_upto=1 --feat_map_wh_scale=2
# tobeat🚩:   "SSIM": 0.8472223281860352, "PSNR": 21.69502830505371, "LPIPS": 0.1668744459748268
# (ours2hT2dF)🧩 where 'dF' means detach 16-dim fetures: --lambda_feat_l1_loss=0.0002 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat)'
#   "SSIM": 0.844883382320404, "PSNR": 21.47751808166504, "LPIPS": 0.16834683964649835

export SCENE=trex; EXP_NAME=${SCENE}_fsgs_ours1; 
CUDA_VISIBLE_DEVICES=2 python train.py  --source_path dataset/nerf_llff_data/$SCENE --model_path output/$EXP_NAME --eval  --n_views 3 --sample_pseudo_interval 1 --load_vgg_img1k_model=1 --use_feat_l1_loss=1 --lambda_feat_l1_loss=1.0 --load_vgg_img1k_upto=1 --feat_map_wh_scale=2
# tobeat🚩:   "SSIM": 0.7984962463378906, "PSNR": 21.35930061340332, "LPIPS": 0.16756346289600646
# (ours2hT2dF)🧩 where 'dF' means detach 16-dim fetures: --lambda_feat_l1_loss=0.0002 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat)'






















# Evaluation

export CUDA_VISIBLE_DEVICES=0
export SCENE=horns; EXP_NAME=${SCENE}_fsgs_ours2hT2doF; 
python render.py --source_path dataset/nerf_llff_data/${SCENE}/  --model_path  output/${EXP_NAME} --iteration 10000
python metrics.py --source_path dataset/nerf_llff_data/${EXP_NAME}/  --model_path  output/${EXP_NAME} --iteration 10000
























# Evaluation
export EXP_NAME=horns_fsgs; SCENE=horns; CUDA_VISIBLE_DEVICES=0
python render.py --source_path dataset/nerf_llff_data/${SCENE}/  --model_path  output/${EXP_NAME} --iteration 10000
python metrics.py --source_path dataset/nerf_llff_data/${EXP_NAME}/  --model_path  output/${EXP_NAME} --iteration 10000




