#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torchmetrics import PearsonCorrCoef
from torchmetrics.functional.regression import pearson_corrcoef
from random import randint
from utils.loss_utils import l1_loss, l1_loss_mask, l2_loss, ssim
from utils.depth_utils import estimate_depth
from gaussian_renderer import render, render_feat, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from lpipsPyTorch import lpips
import torchvision
from torchvision import transforms
from PIL import Image

def tv_loss(img, tv_weight):
    """
    Compute total variation loss.

    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.

    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    w_variance = torch.sum(torch.pow(img[:,:,:,:-1] - img[:,:,:,1:], 2))
    h_variance = torch.sum(torch.pow(img[:,:,:-1,:] - img[:,:,1:,:], 2))
    loss = tv_weight * (h_variance + w_variance)
    return loss

def tv_loss_2(img, img_gt, loss_calc):
    w_variance = torch.std(img[:,:,:-1] - img[:,:,1:], dim=0)
    h_variance = torch.std(img[:,:-1,:] - img[:,1:,:], dim=0)
    w_variance_gt = torch.std(img_gt[:,:,:-1] - img_gt[:,:,1:], dim=0)
    h_variance_gt = torch.std(img_gt[:,:-1,:] - img_gt[:,1:,:], dim=0)
    loss = loss_calc(w_variance_gt, w_variance) + loss_calc(h_variance_gt, h_variance)
    return loss

def tv_loss_3(img, img_gt, loss_calc):
    w_variance = img[:,:,:-1] - img[:,:,1:]
    h_variance = img[:,:-1,:] - img[:,1:,:]
    w_variance_gt = img_gt[:,:,:-1] - img_gt[:,:,1:]
    h_variance_gt = img_gt[:,:-1,:] - img_gt[:,1:,:]
    loss = loss_calc(w_variance_gt, w_variance) + loss_calc(h_variance_gt, h_variance)
    return loss


def training(dataset, opt, pipe, args):
    testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from = args.test_iterations, \
            args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, shuffle=False)
    gaussians.training_setup(opt)
    l1_loss_torch = torch.nn.L1Loss()
    if args.load_vgg_img1k_model:
        vgg16 = torchvision.models.vgg16(weights='IMAGENET1K_V1').to("cuda")
        feat_model = vgg16.features[:args.load_vgg_img1k_upto]
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        preprocess_mean_std = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # preprocess_gt_scale = transforms.Compose([
        # ])
    if checkpoint:
        print(1111/0)
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    
    if os.args.only_render_feat:
        render = render_feat


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

    viewpoint_stack, pseudo_stack = None, None
    ema_loss_for_log = 0.0
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()  

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if os.args.load_vgg_img1k_model and not os.args.only_render_feat:
            render_pkg_feat = render_feat(viewpoint_cam, gaussians, pipe, background)
            render_pkg['feature_map'] = render_pkg_feat['feature_map'] 

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 =  l1_loss_mask(image, gt_image)
        loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)))

        feat_gt = None
        if args.load_vgg_img1k_model:
            assert "feature_map" in render_pkg.keys()
            feature_map = render_pkg["feature_map"]

            assert os.args.use_vgg_preprocess + os.args.use_vgg_preprocess_mean_std <= 1
            if os.args.use_vgg_preprocess:
                img_feed = preprocess(image)
            elif os.args.use_vgg_preprocess_mean_std:
                img_feed = preprocess_mean_std(image)
            else:
                img_feed = image
            feat_gt = feat_model(img_feed).detach()
            
            # if os.args.semantic_feature_mode == "59-feat":
            feature_map = feature_map[:feat_gt.shape[0], :,:]

            if type(feat_model[-1]).__name__ in ['ReLU']:
                # assert 'relu' == os.args.transformation_layer_mode.split('+')[-1]
                feature_map = feature_map.relu()

        if args.use_feat_l1_loss:
            assert args.load_vgg_img1k_model
            # size =[int(viewpoint_cam.image_height / os.args.feat_map_wh_scale), int(viewpoint_cam.image_width  / os.args.feat_map_wh_scale )]
            # feat_gt = torch.nn.functional.interpolate(feat_gt[None,...], size=size, mode='bilinear')[0]
            assert os.args.feat_map_wh_scale == 1 

            if not(feat_gt.shape[1:] == feature_map.shape[1:]):
                size = feat_gt.shape[1:]
                feature_map = torch.nn.functional.interpolate(feature_map[None,...], size=size, mode='bilinear')[0]

            assert feat_gt.shape == feature_map.shape, (feat_gt.shape, feature_map.shape)

            l1_feat = l1_loss_torch(feature_map, feat_gt)
            # print(l1_feat)
            loss += l1_feat * args.lambda_feat_l1_loss

        if args.use_scaling_std_loss:
            target_std = torch.Tensor([args.target_scaling_std]).to("cuda")
            loss_std = l1_loss_torch(target_std, scene.gaussians.get_scaling.std(dim=0).mean())
            loss += loss_std * args.lambda_scaling_std
        
        if args.use_total_var_diff_loss:
            tv_loss_gt = tv_loss(gt_image[None, ...], args.tv_weight)
            tv_loss_pred = tv_loss(image[None, ...], args.tv_weight)
            tv_diff_loss = l1_loss_torch(tv_loss_gt, tv_loss_pred) 
            loss += tv_diff_loss * args.lambda_tv_diff
        
        if args.use_wh_var_diff_loss:
            assert args.wh_var_diff_loss_mode in ['tv_loss_2', 'tv_loss_3']
            tv_loss_func = eval(args.wh_var_diff_loss_mode)
            wh_var_diff_loss = tv_loss_func(image, gt_image, l1_loss_torch)
            loss += wh_var_diff_loss * args.lambda_wh_var_diff 

            
        rendered_depth = render_pkg["depth"][0]
        midas_depth = torch.tensor(viewpoint_cam.depth_image).cuda()
        rendered_depth = rendered_depth.reshape(-1, 1)
        midas_depth = midas_depth.reshape(-1, 1)

        depth_loss = min(
                        (1 - pearson_corrcoef( - midas_depth, rendered_depth)),
                        (1 - pearson_corrcoef(1 / (midas_depth + 200.), rendered_depth))
        )
        loss += args.depth_weight * depth_loss

        if iteration > args.end_sample_pseudo:
            args.depth_weight = 0.001

        if iteration > args.end_feat_reg:
            args.lambda_feat_l1_loss = args.lambda_feat_l1_after_end_feat_reg

        if iteration % args.sample_pseudo_interval == 0 and iteration > args.start_sample_pseudo and iteration < args.end_sample_pseudo:
            if not pseudo_stack:
                pseudo_stack = scene.getPseudoCameras().copy()
            pseudo_cam = pseudo_stack.pop(randint(0, len(pseudo_stack) - 1))

            render_pkg_pseudo = render(pseudo_cam, gaussians, pipe, background)
            rendered_depth_pseudo = render_pkg_pseudo["depth"][0]
            midas_depth_pseudo = estimate_depth(render_pkg_pseudo["render"], mode='train')

            rendered_depth_pseudo = rendered_depth_pseudo.reshape(-1, 1)
            midas_depth_pseudo = midas_depth_pseudo.reshape(-1, 1)
            depth_loss_pseudo = (1 - pearson_corrcoef(rendered_depth_pseudo, -midas_depth_pseudo)).mean()

            if torch.isnan(depth_loss_pseudo).sum() == 0:
                loss_scale = min((iteration - args.start_sample_pseudo) / 500., 1)
                loss += loss_scale * args.depth_pseudo_weight * depth_loss_pseudo


        loss.backward()
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                            testing_iterations, scene, render, (pipe, background))

            if iteration > first_iter and (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration > first_iter and (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration),
                           scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # Densification
            if  iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_threshold, scene.cameras_extent, size_threshold, iteration)


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            gaussians.update_learning_rate(iteration)
            if (iteration - args.start_sample_pseudo - 1) % opt.opacity_reset_interval == 0 and \
                    iteration > args.start_sample_pseudo:
                gaussians.reset_opacity()


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    # os.makedirs(args.model_path, exist_ok = True)
    exist_ok = 'debug' in args.model_path
    os.makedirs(args.model_path, exist_ok = exist_ok)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer



def training_report(tb_writer, iteration, Ll1, loss, l1_loss, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        # tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test, psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 8):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()

                    _mask = None
                    _psnr = psnr(image, gt_image, _mask).mean().double()
                    _ssim = ssim(image, gt_image, _mask).mean().double()
                    _lpips = lpips(image, gt_image, _mask, net_type='vgg')
                    psnr_test += _psnr
                    ssim_test += _ssim
                    lpips_test += _lpips
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {} ".format(
                    iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)

    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10_00, 20_00, 30_00, 50_00, 10_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[50_00, 10_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[50_00, 10_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--train_bg", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    os.args = args
    args.save_iterations.append(args.iterations)

    print(args.test_iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args)

    # All done
    print("\nTraining complete.")