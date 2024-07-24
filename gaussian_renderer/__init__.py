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
import matplotlib.pyplot as plt
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_gaussian_rasterization_feat import GaussianRasterizationSettings as GaussianRasterizationSettings_feat, GaussianRasterizer as GaussianRasterizer_feat
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
import os


def render(viewpoint_camera, pc, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0,
           override_color = None, white_bg = False):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if min(pc.bg_color.shape) != 0:
        bg_color = torch.tensor([0., 0., 0.]).cuda()

    confidence = pc.confidence if pipe.use_confidence else torch.ones_like(pc.confidence)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg = bg_color, #torch.tensor([1., 1., 1.]).cuda() if white_bg else torch.tensor([0., 0., 0.]).cuda(), #bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        confidence=confidence
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color


    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, depth, alpha = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # rendered_image_list, depth_list, alpha_list = [], [], []
    # for i in range(5):
    #     rendered_image, radii, depth, alpha = rasterizer(
    #         means3D=means3D,
    #         means2D=means2D,
    #         shs=shs,
    #         colors_precomp=colors_precomp,
    #         opacities=opacity,
    #         scales=scales,
    #         rotations=rotations,
    #         cov3D_precomp=cov3D_precomp)
    #     rendered_image_list.append(rendered_image)
    #     depth_list.append(depth)
    #     alpha_list.append(alpha)
    # def mean1(t):
    #     return torch.mean(torch.stack(t), 0)
    # rendered_image, depth, alpha = mean1(rendered_image_list), mean1(depth_list), mean1(alpha_list)

    if min(pc.bg_color.shape) != 0:
        rendered_image = rendered_image + (1 - alpha) * torch.sigmoid(pc.bg_color)  # torch.ones((3, 1, 1)).cuda()


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth": depth}




def render_feat(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, ret_pts=False):
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    var_loss = torch.zeros((1, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    try:
        var_loss.retain_grad()
    except:
        pass
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings_feat = GaussianRasterizationSettings_feat(
        image_height=int(viewpoint_camera.image_height / os.args.feat_map_wh_scale),
        image_width=int(viewpoint_camera.image_width  / os.args.feat_map_wh_scale ),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        # beta = pipe.beta,
        debug=pipe.debug,
        # ret_pts=ret_pts
    )


    rasterizer_feat = GaussianRasterizer_feat(raster_settings=raster_settings_feat)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    
    # >>>>>>>>>>>>
    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    feature_map = None
    # if os.args.load_vgg_img1k_model:
    if os.args.semantic_feature_mode == '59':
        semantic_feature = torch.concat([pc.get_xyz, pc.get_opacity, pc.get_rotation, pc.get_scaling, pc.get_features.view(pc.get_features.shape[0], -1)], dim=1)
    elif os.args.semantic_feature_mode == '59-feat':
        semantic_feature = torch.concat([pc.get_xyz, pc.get_opacity, pc.get_rotation, pc.get_scaling], dim=1)
    elif os.args.semantic_feature_mode == '59dt(feat)':
        semantic_feature = torch.concat([pc.get_xyz, pc.get_opacity, pc.get_rotation, pc.get_scaling, pc.get_features.detach().view(pc.get_features.shape[0], -1)], dim=1)
    elif os.args.semantic_feature_mode == '59dt(feat+o)':
        semantic_feature = torch.concat([pc.get_xyz, pc.get_opacity.detach(), pc.get_rotation, pc.get_scaling, pc.get_features.detach().view(pc.get_features.shape[0], -1)], dim=1)
    elif os.args.semantic_feature_mode == '59-(feat+o)':
        semantic_feature = torch.concat([pc.get_xyz, pc.get_rotation, pc.get_scaling], dim=1)
    elif os.args.semantic_feature_mode == '59dt(feat+o+r)':
        semantic_feature = torch.concat([pc.get_xyz, pc.get_opacity.detach(), pc.get_rotation.detach(), pc.get_scaling, pc.get_features.detach().view(pc.get_features.shape[0], -1)], dim=1)
    else:
        raise NotImplementedError()

    semantic_feature = pc.transformation_layer(semantic_feature)
    # print("pc.get_xyz.shape", pc.get_xyz.shape)
    # print("semantic_feature", semantic_feature.shape)
    # print("semantic_feature.norm", semantic_feature.norm(dim=-1))
    semantic_feature = semantic_feature[:,None,:]
    # print(2/0)
    rendered_image, feature_map, radii = rasterizer_feat(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        semantic_feature = semantic_feature, 
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    # <<<<<<<<<<<<
    if os.args.transformation_post_layer_mode != '':
        feature_map = pc.transformation_layer_post(feature_map.permute(1,2,0)).permute(2,0,1)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            'feature_map': feature_map}
