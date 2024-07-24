

# Baseline
export SCENE=bicycle; EXP_NAME=${SCENE}_fsgs;
CUDA_VISIBLE_DEVICES=1 python train.py  --source_path dataset/mipnerf360/$SCENE --model_path output/$EXP_NAME --eval  --n_views 12 --sample_pseudo_interval 1 
#   "SSIM": 0.3306053876876831, "PSNR": 17.016542434692383, "LPIPS": 0.5502325665950775
# (fsgs_cp1): copy of _fsgs: 
#   "SSIM": 0.33575183153152466, "PSNR": 17.18061065673828, "LPIPS": 0.5522061181068421
# (_fsgso) --sample_pseudo_interval=<default> --depth_pseudo_weight=0.03
#   "SSIM": 0.37082672119140625🚩, "PSNR": 18.56578826904297, "LPIPS": 0.5254740476608276
# --------------------------- 24 views ---------------------------
# (_fsgs24): --n_views=24
#   "SSIM": 0.47258681058883667, "PSNR": 20.802881240844727, "LPIPS": 0.4881946063041687
# (_fsgs24o): where 'o' for 'origin': --n_views=24 --sample_pseudo_interval=<default> --depth_pseudo_weight=0.03
#   "SSIM": 0.5090504288673401, "PSNR": 21.454652786254883, "LPIPS": 0.4480331766605377
# (_fsgs24o4x): --n_views=24 --sample_pseudo_interval=<default> --depth_pseudo_weight=0.03 --_images=images_4



export SCENE=bonsai; EXP_NAME=${SCENE}_fsgso;
CUDA_VISIBLE_DEVICES=1 python train.py  --source_path dataset/mipnerf360/$SCENE --model_path output/$EXP_NAME --eval  --n_views 12 --depth_pseudo_weight=0.03
# (_fsgso) --sample_pseudo_interval=<default> --depth_pseudo_weight=0.03
#   "SSIM": 0.6248692870140076🚩, "PSNR": 18.900426864624023, "LPIPS": 0.31986812179958496
# (ourso2hT2doF) where 'doF' means detach the opacity feature and the 16-dim fetures: --lambda_feat_l1_loss=0.0002 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat+o)' --sample_pseudo_interval=<default> --depth_pseudo_weight=0.03 
#   "SSIM": 0.6260617971420288, "PSNR": 19.041866302490234, "LPIPS": 0.3215903541123545


export SCENE=counter; EXP_NAME=${SCENE}_fsgso;
CUDA_VISIBLE_DEVICES=1 python train.py  --source_path dataset/mipnerf360/$SCENE --model_path output/$EXP_NAME --eval  --n_views 12 --depth_pseudo_weight=0.03 
# (_fsgso) --sample_pseudo_interval=<default> --depth_pseudo_weight=0.03
#   "SSIM": 0.5907432436943054🚩, "PSNR": 18.6829833984375, "LPIPS": 0.34003566776712735


export SCENE=garden; EXP_NAME=${SCENE}_fsgso;
CUDA_VISIBLE_DEVICES=1 python train.py  --source_path dataset/mipnerf360/$SCENE --model_path output/$EXP_NAME --eval  --n_views 12  --depth_pseudo_weight=0.03
# (_fsgso) --sample_pseudo_interval=<default> --depth_pseudo_weight=0.03
#   "SSIM": 0.5651319026947021🚩, "PSNR": 19.8405818939209, "LPIPS": 0.3576882767180602
 
 
export SCENE=kitchen; EXP_NAME=${SCENE}_fsgso;
CUDA_VISIBLE_DEVICES=2 python train.py  --source_path dataset/mipnerf360/$SCENE --model_path output/$EXP_NAME --eval  --n_views 12 --depth_pseudo_weight=0.03
# (_fsgso) --sample_pseudo_interval=<default> --depth_pseudo_weight=0.03
#   "SSIM": 0.7269503474235535, "PSNR": 19.968637466430664, "LPIPS": 0.2305670440196991

export SCENE=room; EXP_NAME=${SCENE}_fsgso;
CUDA_VISIBLE_DEVICES=2 python train.py  --source_path dataset/mipnerf360/$SCENE --model_path output/$EXP_NAME --eval  --n_views 12 --depth_pseudo_weight=0.03 
# (_fsgso) --sample_pseudo_interval=<default> --depth_pseudo_weight=0.03
#   "SSIM": 0.7213128209114075, "PSNR": 20.803409576416016, "LPIPS": 0.2643714436353781

export SCENE=stump; EXP_NAME=${SCENE}_fsgso;
CUDA_VISIBLE_DEVICES=2 python train.py  --source_path dataset/mipnerf360/$SCENE --model_path output/$EXP_NAME --eval  --n_views 12 --depth_pseudo_weight=0.03 
# (_fsgso) --sample_pseudo_interval=<default> --depth_pseudo_weight=0.03
#   "SSIM": 0.19364899396896362, "PSNR": 15.920699119567871, "LPIPS": 0.6119224801659584


# Ours
export SCENE=bicycle; EXP_NAME=${SCENE}_fsgs_ours2f;
CUDA_VISIBLE_DEVICES=2 python train.py  --source_path dataset/mipnerf360/$SCENE --model_path output/$EXP_NAME --eval  --n_views 12 --sample_pseudo_interval 1 --load_vgg_img1k_model=1 --use_feat_l1_loss=1 --lambda_feat_l1_loss=0.002 --load_vgg_img1k_upto=1 --feat_map_wh_scale=1
# (ours2f) --lambda_feat_l1_loss=0.002 --feat_map_wh_scale=1
#   "SSIM": 0.3342686593532562, "PSNR": 17.08977508544922, "LPIPS": 0.5749959993362427
# (ours2fxF): --lambda_feat_l1_loss=0.002 --feat_map_wh_scale=1 
#   "SSIM": 0.338468998670578, "PSNR": 17.318984985351562, "LPIPS": 0.5502177560329438
# (ourso2hT2xF) where 'dF' means detach 16-dim fetures: --lambda_feat_l1_loss=0.0002 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59-feat' --sample_pseudo_interval=<default> --depth_pseudo_weight=0.03 
#   "SSIM": 0.3673558831214905, "PSNR": 18.32978057861328, "LPIPS": 0.5197918224334717
# (ourso2hT2dF) where 'dF' means detach 16-dim fetures: --lambda_feat_l1_loss=0.0002 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat)' --sample_pseudo_interval=<default> --depth_pseudo_weight=0.03 
#   "SSIM": 0.37484514713287354, "PSNR": 18.520458221435547, "LPIPS": 0.524728752374649
# (ourso2iT2dF) where 'dF' means detach 16-dim fetures: --lambda_feat_l1_loss=0.0005 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat)' --sample_pseudo_interval=<default> --depth_pseudo_weight=0.03 
#   "SSIM": 0.37084391713142395, "PSNR": 18.3936767578125, "LPIPS": 0.5282290470600128
# (ours2jT2dF) where 'dF' means detach 16-dim fetures: --lambda_feat_l1_loss=0.0001 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat)' --sample_pseudo_interval=<default> --depth_pseudo_weight=0.03 
#   "SSIM": 0.368602454662323, "PSNR": 18.346660614013672, "LPIPS": 0.5168921852111816
# (ourso2hT2dFP) where 'P' means using preprocess transformation: --lambda_feat_l1_loss=0.0002 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat)' --sample_pseudo_interval=<default> --depth_pseudo_weight=0.03 --use_vgg_preprocess=1
#   "SSIM": 0.3718300759792328, "PSNR": 18.689743041992188, "LPIPS": 0.5261470258235932
# (ourso2hT2P) : --lambda_feat_l1_loss=0.0002 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59' --sample_pseudo_interval=<default> --depth_pseudo_weight=0.03 --use_vgg_preprocess=1
#   "SSIM": 0.3686122000217438, "PSNR": 18.284433364868164, "LPIPS": 0.5375549554824829
# (ourso2hT2P1) where P1 means using preprocess mode 1, i.e., --use_vgg_preprocess_mean_std=1 : --lambda_feat_l1_loss=0.0002 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59' --sample_pseudo_interval=<default> --depth_pseudo_weight=0.03 --use_vgg_preprocess_mean_std=1
#   "SSIM": 0.36758896708488464, "PSNR": 17.955886840820312, "LPIPS": 0.5459054815769195
# (ourso2hT2doF) where 'doF' means detach the opacity feature and the 16-dim fetures: --lambda_feat_l1_loss=0.0002 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat+o)' --sample_pseudo_interval=<default> --depth_pseudo_weight=0.03 
#   "SSIM": 0.37579014897346497👍, "PSNR": 18.65152359008789, "LPIPS": 0.5261498546600342
# (ourso2hT2xoF) where 'xoF' means remove both the opacity feature and the 16-dim fetures: --lambda_feat_l1_loss=0.0002 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59-(feat+o)' --sample_pseudo_interval=<default> --depth_pseudo_weight=0.03 
#   "SSIM": 0.37472498416900635, "PSNR": 18.580564498901367, "LPIPS": 0.5278502035140992
# (ourso2hT2dorF) where 'dorF' means detach the opacity feature, rotation and the 16-dim fetures: --lambda_feat_l1_loss=0.0002 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat+o+r)' --sample_pseudo_interval=<default> --depth_pseudo_weight=0.03 
#   "SSIM": 0.3733134865760803, "PSNR": 18.64729118347168, "LPIPS": 0.5231177604198456
# (ourso2hT2doFStd) where 'Std' means using scaling std loss: --lambda_feat_l1_loss=0.0002 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat+o)' --sample_pseudo_interval=<default> --depth_pseudo_weight=0.03 --use_scaling_std_loss=1
#   "SSIM": 0.3745132088661194, "PSNR": 18.681560516357422, "LPIPS": 0.5255082845687866
# (ourso2hT2doFStd1) where 'Std' means using scaling std loss: --lambda_feat_l1_loss=0.0002 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat+o)' --sample_pseudo_interval=<default> --depth_pseudo_weight=0.03 --use_scaling_std_loss=1 --target_scaling_std=0.1
#   "SSIM": 0.36465585231781006, "PSNR": 18.2927188873291, "LPIPS": 0.5220027697086335
# (ourso2hT2doFTV) where 'TV' means using total variance loss: --lambda_feat_l1_loss=0.0002 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat+o)' --sample_pseudo_interval=<default> --depth_pseudo_weight=0.03 --use_total_var_diff_loss=1
#   "SSIM": 0.3657301366329193, "PSNR": 18.19867515563965, "LPIPS": 0.5246159744262695
# (ourso2hT2doFTV1): --lambda_feat_l1_loss=0.0002 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat+o)' --sample_pseudo_interval=<default> --depth_pseudo_weight=0.03 --use_total_var_diff_loss=1 --tv_weight=1e-5
#   "SSIM": 0.28659263253211975, "PSNR": 16.049379348754883, "LPIPS": 0.5818155264854431
# (ourso2hT2doFTV2): --lambda_feat_l1_loss=0.0002 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat+o)' --sample_pseudo_interval=<default> --depth_pseudo_weight=0.03 --use_total_var_diff_loss=1 --tv_weight=5e-6
#   "SSIM": 0.31370314955711365, "PSNR": 16.6475772857666, "LPIPS": 0.5648824787139892
# (ourso2hT2doFTV3): --lambda_feat_l1_loss=0.0002 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat+o)' --sample_pseudo_interval=<default> --depth_pseudo_weight=0.03 --use_total_var_diff_loss=1 --tv_weight=5e-7
#   "SSIM": 0.36205509305000305, "PSNR": 18.071138381958008, "LPIPS": 0.5167830288410187
# (ourso2hT2doFTV4): --lambda_feat_l1_loss=0.0002 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat+o)' --sample_pseudo_interval=<default> --depth_pseudo_weight=0.03 --use_total_var_diff_loss=1 --tv_weight=1e-7
#   "SSIM": 0.3645131289958954, "PSNR": 18.252710342407227, "LPIPS": 0.5232429599761963
# (ourso2hT2doFTV5): --lambda_feat_l1_loss=0.0002 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat+o)' --sample_pseudo_interval=<default> --depth_pseudo_weight=0.03 --use_total_var_diff_loss=1 --tv_weight=1e-8
#   "SSIM": 0.3706204295158386, "PSNR": 18.377397537231445, "LPIPS": 0.520767115354538
# (ourso2hT2doFTVa): where 'TVa' means using "--use_wh_var_diff_loss";  --lambda_feat_l1_loss=0.0002 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat+o)' --sample_pseudo_interval=<default> --depth_pseudo_weight=0.03 --use_wh_var_diff_loss=1 
#   "SSIM": 0.37537822127342224, "PSNR": 18.560396194458008, "LPIPS": 0.5245625650882721
# (ourso2hT2doFTVb): where 'TVb' means using "--use_wh_var_diff_loss", "--lambda_wh_var_diff=0.5" and "--wh_var_diff_loss_mode=tv_loss_3";  --lambda_feat_l1_loss=0.0002 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat+o)' --sample_pseudo_interval=<default> --depth_pseudo_weight=0.03 --use_wh_var_diff_loss=1 --wh_var_diff_loss_mode=tv_loss_3 --lambda_wh_var_diff=0.5
#   "SSIM": 0.35887759923934937, "PSNR": 17.967056274414062, "LPIPS": 0.5175157856941223
# (ourso2hT2doFTVb0): where 'TVb0' means using "--use_wh_var_diff_loss" and "--wh_var_diff_loss_mode=tv_loss_3";  --lambda_feat_l1_loss=0.0002 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat+o)' --sample_pseudo_interval=<default> --depth_pseudo_weight=0.03 --use_wh_var_diff_loss=1 --wh_var_diff_loss_mode=tv_loss_3 
# (ourso2hT2doFTVb2): where 'TVb0' means using "--use_wh_var_diff_loss" and "--wh_var_diff_loss_mode=tv_loss_3";  --lambda_feat_l1_loss=0.0002 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat+o)' --sample_pseudo_interval=<default> --depth_pseudo_weight=0.03 --use_wh_var_diff_loss=1 --wh_var_diff_loss_mode=tv_loss_3 --lambda_wh_var_diff=0.05 
#   "SSIM": 0.3730621039867401, "PSNR": 18.41925621032715, "LPIPS": 0.5233383178710938
# (ourso2hT2doFTVb3): where 'TVb0' means using "--use_wh_var_diff_loss" and "--wh_var_diff_loss_mode=tv_loss_3";  --lambda_feat_l1_loss=0.0002 --feat_map_wh_scale=1 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat+o)' --sample_pseudo_interval=<default> --depth_pseudo_weight=0.03 --use_wh_var_diff_loss=1 --wh_var_diff_loss_mode=tv_loss_3 --lambda_wh_var_diff=0.005 
#   "SSIM": 0.3689563274383545, "PSNR": 18.3098087310791, "LPIPS": 0.5206359875202179



export SCENE=counter; EXP_NAME=${SCENE}_fsgs_ourso2hT2doF;
CUDA_VISIBLE_DEVICES=2 python train.py  --source_path dataset/mipnerf360/$SCENE --model_path output/$EXP_NAME --eval  --n_views 12 --load_vgg_img1k_model=1 --use_feat_l1_loss=1 --lambda_feat_l1_loss=0.0002 --load_vgg_img1k_upto=1 --feat_map_wh_scale=1 --depth_pseudo_weight=0.03 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat+o)'
# To beat:  "SSIM": 0.5907432436943054🚩, "PSNR": 18.6829833984375, "LPIPS": 0.34003566776712735
#   "SSIM": 0.5915676355361938, "PSNR": 18.47127342224121, "LPIPS": 0.34216593677798907

export SCENE=garden; EXP_NAME=${SCENE}_fsgs_ourso2hT2doF;
CUDA_VISIBLE_DEVICES=1 python train.py  --source_path dataset/mipnerf360/$SCENE --model_path output/$EXP_NAME --eval  --n_views 12 --load_vgg_img1k_model=1 --use_feat_l1_loss=1 --lambda_feat_l1_loss=0.0002 --load_vgg_img1k_upto=1 --feat_map_wh_scale=1 --depth_pseudo_weight=0.03 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat+o)'

export SCENE=kitchen; EXP_NAME=${SCENE}_fsgs_ourso2hT2doF;
CUDA_VISIBLE_DEVICES=1 python train.py  --source_path dataset/mipnerf360/$SCENE --model_path output/$EXP_NAME --eval  --n_views 12 --load_vgg_img1k_model=1 --use_feat_l1_loss=1 --lambda_feat_l1_loss=0.0002 --load_vgg_img1k_upto=1 --feat_map_wh_scale=1 --depth_pseudo_weight=0.03 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat+o)'

export SCENE=room; EXP_NAME=${SCENE}_fsgs_ourso2hT2doF;
CUDA_VISIBLE_DEVICES=1 python train.py  --source_path dataset/mipnerf360/$SCENE --model_path output/$EXP_NAME --eval  --n_views 12 --load_vgg_img1k_model=1 --use_feat_l1_loss=1 --lambda_feat_l1_loss=0.0002 --load_vgg_img1k_upto=1 --feat_map_wh_scale=1 --depth_pseudo_weight=0.03 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat+o)'

export SCENE=stump; EXP_NAME=${SCENE}_fsgs_ourso2hT2doF;
CUDA_VISIBLE_DEVICES=1 python train.py  --source_path dataset/mipnerf360/$SCENE --model_path output/$EXP_NAME --eval  --n_views 12 --load_vgg_img1k_model=1 --use_feat_l1_loss=1 --lambda_feat_l1_loss=0.0002 --load_vgg_img1k_upto=1 --feat_map_wh_scale=1 --depth_pseudo_weight=0.03 --transformation_layer_mode=linear+relu+linear --semantic_feature_mode='59dt(feat+o)'









# --------------------------- 24 views ---------------------------

# (_fsgs24_ours2f) --lambda_feat_l1_loss=0.002 --feat_map_wh_scale=1 --n_views 24 
#   "SSIM": 0.46175628900527954, "PSNR": 20.123865127563477, "LPIPS": 0.5045112049579621
# (_fsgs24_ours2fxF) --lambda_feat_l1_loss=0.002 --feat_map_wh_scale=1 --n_views 24 --semantic_feature_mode=59-feat
#   "SSIM": 0.4656333923339844, "PSNR": 20.426233291625977, "LPIPS": 0.4956676304340363
# (_fsgs24o4x): --n_views=24 --sample_pseudo_interval=<default> --depth_pseudo_weight=0.03 --images=images_4
#   "SSIM": 0.46054184436798096, "PSNR": 20.8986759185791, "LPIPS": 0.5059874737262726




# Evaluation
export CUDA_VISIBLE_DEVICES=3
export SCENE=bicycle; EXP_NAME=${SCENE}_ourso2hT2dF; 
python render.py --source_path dataset/mipnerf360/${SCENE}/  --model_path  output/${EXP_NAME} --iteration 10000
python metrics.py --source_path dataset/mipnerf360/${EXP_NAME}/  --model_path  output/${EXP_NAME} --iteration 10000






# Evaluation for 4x resolution
export CUDA_VISIBLE_DEVICES=0
export SCENE=bicycle; EXP_NAME=${SCENE}_fsgs_ours2fxF; 
python render.py --source_path dataset/mipnerf360/${SCENE}/  --model_path  output/${EXP_NAME} --iteration 10000 --images=images_4
python metrics.py --source_path dataset/mipnerf360/${EXP_NAME}/  --model_path  output/${EXP_NAME} --iteration 10000 --images=images_4





export CUDA_VISIBLE_DEVICES=2
export SCENE=bicycle; EXP_NAME=${SCENE}_fsgs24o4x;
python render.py --source_path dataset/mipnerf360/$SCENE --model_path  output/$EXP_NAME --iteration 10000  --video  --fps 30










