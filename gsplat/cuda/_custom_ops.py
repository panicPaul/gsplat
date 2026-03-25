"""Frontend custom ops for gsplat.

The C++/CUDA backend exposes low-level dispatcher ops such as `*_fwd` and
`*_bwd`. This module defines the high-level differentiable frontend ops and
attaches autograd and fake tensor behavior on top of those low-level kernels.
"""

import math
from typing import Any

import torch
from torch import Tensor


def _has_custom_classes() -> bool:
    try:
        from ._backend import _C
    except Exception:
        return False
    if _C is None:
        return False
    try:
        torch.classes.gsplat.UnscentedTransformParameters
        torch.classes.gsplat.FThetaCameraDistortionParameters
    except RuntimeError:
        return False
    return True


def _make_lazy_cuda_obj(name: str) -> Any:
    from ._backend import _C

    if _C is None:
        raise RuntimeError(
            "gsplat CUDA extension is not available (not built or failed to load). "
            f"Cannot access '{name}'."
        )
    obj = _C
    for name_split in name.split("."):
        obj = getattr(obj, name_split)
    return obj


def _camera_model_type(camera_model: str) -> Any:
    return _make_lazy_cuda_obj(f"CameraModelType.{camera_model.upper()}")


def _dynamic_size() -> int:
    return torch.library.get_ctx().new_dynamic_size()


def _int_output(device: torch.device, shape: tuple[Any, ...]) -> Tensor:
    return torch.empty(shape, device=device, dtype=torch.int32)


@torch.library.custom_op(
    "gsplat::quat_scale_to_covar_preci",
    mutates_args=(),
    device_types="cuda",
    schema="(Tensor quats, Tensor scales, bool compute_covar=True, bool compute_preci=True, bool triu=False) -> (Tensor?, Tensor?)",
)
def quat_scale_to_covar_preci(
    quats: Tensor,
    scales: Tensor,
    compute_covar: bool = True,
    compute_preci: bool = True,
    triu: bool = False,
) -> tuple[Tensor | None, Tensor | None]:
    """Convert quaternions and scales to covariance and precision tensors.

    Expected shapes:
    - `quats`: `[..., 4]`
    - `scales`: `[..., 3]`

    Returns:
    - covariance tensor with shape `[..., 6]` when `triu=True`, else `[..., 3, 3]`
    - precision tensor with shape `[..., 6]` when `triu=True`, else `[..., 3, 3]`
    """
    covars, precis = torch.ops.gsplat.quat_scale_to_covar_preci_fwd(
        quats, scales, compute_covar, compute_preci, triu
    )
    return covars if compute_covar else None, precis if compute_preci else None


@torch.library.custom_op(
    "gsplat::spherical_harmonics",
    mutates_args=(),
    device_types="cuda",
    schema="(int sh_degree, Tensor dirs, Tensor coeffs, Tensor? masks=None) -> Tensor",
)
def spherical_harmonics(
    sh_degree: int, dirs: Tensor, coeffs: Tensor, masks: Tensor | None = None
) -> Tensor:
    """Evaluate spherical harmonics colors.

    Expected shapes:
    - `dirs`: `[..., 3]`
    - `coeffs`: `[..., K, 3]`
    - `masks`: `[...]` when provided

    Returns:
    - colors with shape `[..., 3]`
    """
    return torch.ops.gsplat.spherical_harmonics_fwd(
        sh_degree, dirs, coeffs, masks
    )


@torch.library.custom_op(
    "gsplat::proj",
    mutates_args=(),
    device_types="cuda",
    schema="(Tensor means, Tensor covars, Tensor Ks, int width, int height, str camera_model='pinhole') -> (Tensor, Tensor)",
)
def proj(
    means: Tensor,
    covars: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    camera_model: str = "pinhole",
) -> tuple[Tensor, Tensor]:
    """Project Gaussians from 3D to 2D.

    Expected shapes:
    - `means`: `[..., C, N, 3]`
    - `covars`: `[..., C, N, 3, 3]`
    - `Ks`: `[..., C, 3, 3]`

    Returns:
    - projected means with shape `[..., C, N, 2]`
    - projected covariances with shape `[..., C, N, 2, 2]`
    """
    return torch.ops.gsplat.projection_ewa_simple_fwd(
        means,
        covars,
        Ks,
        width,
        height,
        _camera_model_type(camera_model),
    )


@torch.library.custom_op(
    "gsplat::fully_fused_projection",
    mutates_args=(),
    device_types="cuda",
    schema="(Tensor means, Tensor? covars, Tensor? quats, Tensor? scales, Tensor viewmats, Tensor Ks, int width, int height, float eps2d=0.3, float near_plane=0.01, float far_plane=10000000000., float radius_clip=0., bool calc_compensations=False, str camera_model='pinhole', Tensor? opacities=None) -> (Tensor, Tensor, Tensor, Tensor, Tensor?)",
)
def fully_fused_projection(
    means: Tensor,
    covars: Tensor | None,
    quats: Tensor | None,
    scales: Tensor | None,
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    calc_compensations: bool = False,
    camera_model: str = "pinhole",
    opacities: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
    """Project 3D Gaussians to image-space using the fused 3DGS path.

    Expected shapes:
    - `means`: `[..., N, 3]`
    - `covars`: `[..., N, 6]` when provided
    - `quats`: `[..., N, 4]` when provided
    - `scales`: `[..., N, 3]` when provided
    - `viewmats`: `[..., C, 4, 4]`
    - `Ks`: `[..., C, 3, 3]`
    - `opacities`: `[..., N]` when provided

    Returns:
    - radii with shape `[..., C, N, 2]`
    - means2d with shape `[..., C, N, 2]`
    - depths with shape `[..., C, N]`
    - conics with shape `[..., C, N, 3]`
    - compensations with shape `[..., C, N]` when enabled, else `None`
    """
    radii, means2d, depths, conics, compensations = (
        torch.ops.gsplat.projection_ewa_3dgs_fused_fwd(
            means,
            covars,
            quats,
            scales,
            opacities,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            calc_compensations,
            _camera_model_type(camera_model),
        )
    )
    return (
        radii,
        means2d,
        depths,
        conics,
        compensations if calc_compensations else None,
    )


@torch.library.custom_op(
    "gsplat::fully_fused_projection_packed",
    mutates_args=(),
    device_types="cuda",
    schema="(Tensor means, Tensor? covars, Tensor? quats, Tensor? scales, Tensor viewmats, Tensor Ks, int width, int height, float eps2d=0.3, float near_plane=0.01, float far_plane=10000000000., float radius_clip=0., bool sparse_grad=False, bool calc_compensations=False, str camera_model='pinhole', Tensor? opacities=None) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor?)",
)
def fully_fused_projection_packed(
    means: Tensor,
    covars: Tensor | None,
    quats: Tensor | None,
    scales: Tensor | None,
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    sparse_grad: bool = False,
    calc_compensations: bool = False,
    camera_model: str = "pinhole",
    opacities: Tensor | None = None,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor | None,
]:
    """Project 3D Gaussians to packed image-space outputs.

    Expected shapes:
    - `means`: `[..., N, 3]`
    - `covars`: `[..., N, 6]` when provided
    - `quats`: `[..., N, 4]` when provided
    - `scales`: `[..., N, 3]` when provided
    - `viewmats`: `[..., C, 4, 4]`
    - `Ks`: `[..., C, 3, 3]`
    - `opacities`: `[..., N]` when provided

    Returns:
    - `batch_ids`: `[nnz]`
    - `camera_ids`: `[nnz]`
    - `gaussian_ids`: `[nnz]`
    - `indptr`: `[B * C + 1]`
    - `radii`: `[nnz, 2]`
    - `means2d`: `[nnz, 2]`
    - `depths`: `[nnz]`
    - `conics`: `[nnz, 3]`
    - `compensations`: `[nnz]` when enabled, else `None`
    """
    (
        indptr,
        batch_ids,
        camera_ids,
        gaussian_ids,
        radii,
        means2d,
        depths,
        conics,
        compensations,
    ) = torch.ops.gsplat.projection_ewa_3dgs_packed_fwd(
        means,
        covars,
        quats,
        scales,
        opacities,
        viewmats,
        Ks,
        width,
        height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        calc_compensations,
        _camera_model_type(camera_model),
    )
    return (
        batch_ids,
        camera_ids,
        gaussian_ids,
        indptr,
        radii,
        means2d,
        depths,
        conics,
        compensations if calc_compensations else None,
    )


@torch.library.custom_op(
    "gsplat::rasterize_to_pixels_extra",
    mutates_args=(),
    device_types="cuda",
    schema="(Tensor means2d, Tensor conics, Tensor colors, Tensor opacities, Tensor? backgrounds, Tensor? masks, int width, int height, int tile_size, Tensor isect_offsets, Tensor flatten_ids, bool absgrad=False) -> (Tensor, Tensor, Tensor, Tensor)",
)
def rasterize_to_pixels_extra(
    means2d: Tensor,
    conics: Tensor,
    colors: Tensor,
    opacities: Tensor,
    backgrounds: Tensor | None,
    masks: Tensor | None,
    width: int,
    height: int,
    tile_size: int,
    isect_offsets: Tensor,
    flatten_ids: Tensor,
    absgrad: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Rasterize 3DGS image-space splats to pixels.

    Expected shapes:
    - `means2d`: `[..., N, 2]` or `[nnz, 2]`
    - `conics`: `[..., N, 3]` or `[nnz, 3]`
    - `colors`: `[..., N, channels]` or `[nnz, channels]`
    - `opacities`: `[..., N]` or `[nnz]`
    - `backgrounds`: `[..., channels]` when provided
    - `masks`: `[..., tile_height, tile_width]` when provided
    - `isect_offsets`: `[..., tile_height, tile_width]`
    - `flatten_ids`: `[n_isects]`

    Returns:
    - `render_colors`: `[..., H, W, channels]`
    - `render_alphas`: `[..., H, W, 1]`
    - `raw_render_alphas`: `[..., H, W, 1]`
    - `last_ids`: `[..., H, W]`
    """
    render_colors, raw_render_alphas, last_ids = (
        torch.ops.gsplat.rasterize_to_pixels_3dgs_fwd(
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
        )
    )
    return (
        render_colors,
        raw_render_alphas.float().clone(),
        raw_render_alphas,
        last_ids,
    )


if _has_custom_classes():

    @torch.library.custom_op(
        "gsplat::fully_fused_projection_with_ut",
        mutates_args=(),
        device_types="cuda",
        schema="(Tensor means, Tensor quats, Tensor scales, Tensor? opacities, Tensor viewmats, Tensor Ks, int width, int height, float eps2d, float near_plane, float far_plane, float radius_clip, bool calc_compensations, str camera_model, __torch__.torch.classes.gsplat.UnscentedTransformParameters ut_params, Tensor? radial_coeffs, Tensor? tangential_coeffs, Tensor? thin_prism_coeffs, __torch__.torch.classes.gsplat.FThetaCameraDistortionParameters ftheta_coeffs, __torch__.torch.classes.gsplat.RowOffsetStructuredSpinningLidarModelParametersExt? lidar_coeffs, __torch__.torch.classes.gsplat.BivariateWindshieldModelParameters? external_distortion_coeffs, int rolling_shutter, Tensor? viewmats_rs, bool global_z_order) -> (Tensor, Tensor, Tensor, Tensor, Tensor?)",
    )
    def fully_fused_projection_with_ut(
        means: Tensor,
        quats: Tensor,
        scales: Tensor,
        opacities: Tensor | None,
        viewmats: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        eps2d: float,
        near_plane: float,
        far_plane: float,
        radius_clip: float,
        calc_compensations: bool,
        camera_model: str,
        ut_params: Any,
        radial_coeffs: Tensor | None,
        tangential_coeffs: Tensor | None,
        thin_prism_coeffs: Tensor | None,
        ftheta_coeffs: Any,
        lidar_coeffs: Any,
        external_distortion_coeffs: Any,
        rolling_shutter: int,
        viewmats_rs: Tensor | None,
        global_z_order: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
        """Project 3DGUT Gaussians with UT, distortion, and rolling shutter.

        Expected shapes:
        - `means`: `[..., N, 3]`
        - `quats`: `[..., N, 4]`
        - `scales`: `[..., N, 3]`
        - `opacities`: `[..., N]` when provided
        - `viewmats`: `[..., C, 4, 4]`
        - `Ks`: `[..., C, 3, 3]`
        - `radial_coeffs`: `[..., C, 6]` or `[..., C, 4]` when provided
        - `tangential_coeffs`: `[..., C, 2]` when provided
        - `thin_prism_coeffs`: `[..., C, 4]` when provided
        - `viewmats_rs`: `[..., C, 4, 4]` when provided

        Returns:
        - radii with shape `[..., C, N, 2]`
        - means2d with shape `[..., C, N, 2]`
        - depths with shape `[..., C, N]`
        - conics with shape `[..., C, N, 3]`
        - compensations with shape `[..., C, N]` when enabled, else `None`
        """
        radii, means2d, depths, conics, compensations = (
            torch.ops.gsplat.projection_ut_3dgs_fused(
                means,
                quats,
                scales,
                opacities,
                viewmats,
                viewmats_rs,
                Ks,
                width,
                height,
                eps2d,
                near_plane,
                far_plane,
                radius_clip,
                calc_compensations,
                _camera_model_type(camera_model),
                global_z_order,
                ut_params,
                rolling_shutter,
                radial_coeffs,
                tangential_coeffs,
                thin_prism_coeffs,
                ftheta_coeffs,
                lidar_coeffs,
                external_distortion_coeffs,
            )
        )
        return (
            radii,
            means2d,
            depths,
            conics,
            compensations if calc_compensations else None,
        )

    @torch.library.custom_op(
        "gsplat::rasterize_to_pixels_eval3d_extra",
        mutates_args=(),
        device_types="cuda",
        schema="(Tensor means, Tensor quats, Tensor scales, Tensor colors, Tensor opacities, Tensor? backgrounds, Tensor? masks, Tensor viewmats, Tensor Ks, int width, int height, int tile_size, Tensor isect_offsets, Tensor flatten_ids, str camera_model, __torch__.torch.classes.gsplat.UnscentedTransformParameters ut_params, Tensor? rays, Tensor? radial_coeffs, Tensor? tangential_coeffs, Tensor? thin_prism_coeffs, __torch__.torch.classes.gsplat.FThetaCameraDistortionParameters ftheta_coeffs, __torch__.torch.classes.gsplat.RowOffsetStructuredSpinningLidarModelParametersExt? lidar_coeffs, __torch__.torch.classes.gsplat.BivariateWindshieldModelParameters? external_distortion_coeffs, int rolling_shutter, Tensor? viewmats_rs, bool return_sample_counts, bool use_hit_distance, bool return_normals) -> (Tensor, Tensor, Tensor, Tensor?, Tensor?)",
    )
    def rasterize_to_pixels_eval3d_extra(
        means: Tensor,
        quats: Tensor,
        scales: Tensor,
        colors: Tensor,
        opacities: Tensor,
        backgrounds: Tensor | None,
        masks: Tensor | None,
        viewmats: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        tile_size: int,
        isect_offsets: Tensor,
        flatten_ids: Tensor,
        camera_model: str,
        ut_params: Any,
        rays: Tensor | None,
        radial_coeffs: Tensor | None,
        tangential_coeffs: Tensor | None,
        thin_prism_coeffs: Tensor | None,
        ftheta_coeffs: Any,
        lidar_coeffs: Any,
        external_distortion_coeffs: Any,
        rolling_shutter: int,
        viewmats_rs: Tensor | None,
        return_sample_counts: bool,
        use_hit_distance: bool,
        return_normals: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
        """Rasterize 3DGUT Gaussians from world-space responses.

        Expected shapes:
        - `means`: `[..., N, 3]`
        - `quats`: `[..., N, 4]`
        - `scales`: `[..., N, 3]`
        - `colors`: `[..., C, N, channels]` or `[nnz, channels]`
        - `opacities`: `[..., C, N]` or `[nnz]`
        - `backgrounds`: `[..., C, channels]` when provided
        - `masks`: `[..., C, tile_height, tile_width]` when provided
        - `viewmats`: `[..., C, 4, 4]`
        - `Ks`: `[..., C, 3, 3]`
        - `rays`: `[..., C, P, 6]` when provided
        - `isect_offsets`: `[..., C, tile_height, tile_width]`
        - `flatten_ids`: `[n_isects]`

        Returns:
        - `render_colors`: `[..., C, H, W, channels]`
        - `render_alphas`: `[..., C, H, W, 1]`
        - `last_ids`: `[..., C, H, W]`
        - `sample_counts`: `[..., C, H, W]` when enabled, else `None`
        - `render_normals`: `[..., C, H, W, 3]` when enabled, else `None`
        """
        batch_dims = means.shape[:-2]
        C = viewmats.shape[-3]
        sample_counts = (
            torch.empty(
                batch_dims + (C, height, width),
                dtype=torch.int32,
                device=means.device,
            )
            if return_sample_counts
            else None
        )
        render_normals = (
            torch.empty(
                batch_dims + (C, height, width, 3),
                dtype=torch.float32,
                device=means.device,
            )
            if return_normals
            else None
        )
        render_colors, render_alphas, last_ids = (
            torch.ops.gsplat.rasterize_to_pixels_from_world_3dgs_fwd(
                means,
                quats,
                scales,
                colors,
                opacities,
                backgrounds,
                masks,
                width,
                height,
                tile_size,
                viewmats,
                viewmats_rs,
                Ks,
                _camera_model_type(camera_model),
                ut_params,
                rolling_shutter,
                rays,
                radial_coeffs,
                tangential_coeffs,
                thin_prism_coeffs,
                ftheta_coeffs,
                lidar_coeffs,
                external_distortion_coeffs,
                isect_offsets,
                flatten_ids,
                use_hit_distance,
                sample_counts,
                render_normals,
            )
        )
        return (
            render_colors,
            render_alphas,
            last_ids,
            sample_counts,
            render_normals,
        )


@torch.library.custom_op(
    "gsplat::fully_fused_projection_2dgs",
    mutates_args=(),
    device_types="cuda",
    schema="(Tensor means, Tensor quats, Tensor scales, Tensor viewmats, Tensor Ks, int width, int height, float eps2d=0.3, float near_plane=0.01, float far_plane=10000000000., float radius_clip=0.) -> (Tensor, Tensor, Tensor, Tensor, Tensor)",
)
def fully_fused_projection_2dgs(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Project 2DGS Gaussians to image-space.

    Expected shapes:
    - `means`: `[..., N, 3]`
    - `quats`: `[..., N, 4]`
    - `scales`: `[..., N, 3]`
    - `viewmats`: `[..., C, 4, 4]`
    - `Ks`: `[..., C, 3, 3]`

    Returns:
    - radii with shape `[..., C, N, 2]`
    - means2d with shape `[..., C, N, 2]`
    - depths with shape `[..., C, N]`
    - ray_transforms with shape `[..., C, N, 3, 3]`
    - normals with shape `[..., C, N, 3]`
    """
    return torch.ops.gsplat.projection_2dgs_fused_fwd(
        means,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
    )


@torch.library.custom_op(
    "gsplat::fully_fused_projection_packed_2dgs",
    mutates_args=(),
    device_types="cuda",
    schema="(Tensor means, Tensor quats, Tensor scales, Tensor viewmats, Tensor Ks, int width, int height, float near_plane=0.01, float far_plane=10000000000., float radius_clip=0., bool sparse_grad=False) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)",
)
def fully_fused_projection_packed_2dgs(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    sparse_grad: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Project 2DGS Gaussians to packed image-space outputs.

    Expected shapes:
    - `means`: `[..., N, 3]`
    - `quats`: `[..., N, 4]`
    - `scales`: `[..., N, 3]`
    - `viewmats`: `[..., C, 4, 4]`
    - `Ks`: `[..., C, 3, 3]`

    Returns:
    - `batch_ids`: `[nnz]`
    - `camera_ids`: `[nnz]`
    - `gaussian_ids`: `[nnz]`
    - `radii`: `[nnz, 2]`
    - `means2d`: `[nnz, 2]`
    - `depths`: `[nnz]`
    - `ray_transforms`: `[nnz, 3, 3]`
    - `normals`: `[nnz, 3]`
    """
    (
        _indptr,
        batch_ids,
        camera_ids,
        gaussian_ids,
        radii,
        means2d,
        depths,
        ray_transforms,
        normals,
    ) = torch.ops.gsplat.projection_2dgs_packed_fwd(
        means,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        near_plane,
        far_plane,
        radius_clip,
    )
    return (
        batch_ids,
        camera_ids,
        gaussian_ids,
        radii,
        means2d,
        depths,
        ray_transforms,
        normals,
    )


@torch.library.custom_op(
    "gsplat::rasterize_to_pixels_2dgs_extra",
    mutates_args=(),
    device_types="cuda",
    schema="(Tensor means2d, Tensor ray_transforms, Tensor colors, Tensor opacities, Tensor normals, Tensor densify, Tensor? backgrounds, Tensor? masks, int width, int height, int tile_size, Tensor isect_offsets, Tensor flatten_ids, bool absgrad=False, bool distloss=False) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)",
)
def rasterize_to_pixels_2dgs_extra(
    means2d: Tensor,
    ray_transforms: Tensor,
    colors: Tensor,
    opacities: Tensor,
    normals: Tensor,
    densify: Tensor,
    backgrounds: Tensor | None,
    masks: Tensor | None,
    width: int,
    height: int,
    tile_size: int,
    isect_offsets: Tensor,
    flatten_ids: Tensor,
    absgrad: bool = False,
    distloss: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Rasterize 2DGS splats to pixels.

    Expected shapes:
    - `means2d`: `[..., N, 2]` or `[nnz, 2]`
    - `ray_transforms`: `[..., N, 3, 3]` or `[nnz, 3, 3]`
    - `colors`: `[..., N, channels]` or `[nnz, channels]`
    - `opacities`: `[..., N]` or `[nnz]`
    - `normals`: `[..., N, 3]` or `[nnz, 3]`
    - `densify`: `[..., N, 2]` or `[nnz, 2]`
    - `backgrounds`: `[..., channels]` when provided
    - `masks`: `[..., tile_height, tile_width]` when provided
    - `isect_offsets`: `[..., tile_height, tile_width]`
    - `flatten_ids`: `[n_isects]`

    Returns:
    - `render_colors`: `[..., H, W, channels]`
    - `render_alphas`: `[..., H, W, 1]`
    - `render_normals`: `[..., H, W, 3]`
    - `render_distort`: `[..., H, W, 1]`
    - `render_median`: `[..., H, W, 1]`
    - `raw_render_alphas`: `[..., H, W, 1]`
    - `last_ids`: `[..., H, W]`
    - `median_ids`: `[..., H, W]`
    """
    (
        render_colors,
        raw_render_alphas,
        render_normals,
        render_distort,
        render_median,
        last_ids,
        median_ids,
    ) = torch.ops.gsplat.rasterize_to_pixels_2dgs_fwd(
        means2d,
        ray_transforms,
        colors,
        opacities,
        normals,
        backgrounds,
        masks,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
    )
    return (
        render_colors,
        raw_render_alphas.float().clone(),
        render_normals,
        render_distort,
        render_median,
        raw_render_alphas,
        last_ids,
        median_ids,
    )


def _quat_scale_to_covar_preci_fake(
    quats: Tensor,
    scales: Tensor,
    compute_covar: bool = True,
    compute_preci: bool = True,
    triu: bool = False,
) -> tuple[Tensor | None, Tensor | None]:
    batch_dims = quats.shape[:-1]
    shape = batch_dims + ((6,) if triu else (3, 3))
    covars = quats.new_empty(shape) if compute_covar else None
    precis = quats.new_empty(shape) if compute_preci else None
    return covars, precis


def _spherical_harmonics_fake(
    sh_degree: int, dirs: Tensor, coeffs: Tensor, masks: Tensor | None = None
) -> Tensor:
    return coeffs.new_empty(coeffs.shape[:-2] + (3,))


def _proj_fake(
    means: Tensor,
    covars: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    camera_model: str = "pinhole",
) -> tuple[Tensor, Tensor]:
    return means.new_empty(means.shape[:-1] + (2,)), covars.new_empty(
        covars.shape[:-2] + (2, 2)
    )


def _fully_fused_projection_fake(
    means: Tensor,
    covars: Tensor | None,
    quats: Tensor | None,
    scales: Tensor | None,
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    calc_compensations: bool = False,
    camera_model: str = "pinhole",
    opacities: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
    batch_dims = means.shape[:-2]
    N = means.shape[-2]
    C = viewmats.shape[-3]
    prefix = batch_dims + (C, N)
    return (
        _int_output(means.device, prefix + (2,)),
        means.new_empty(prefix + (2,)),
        means.new_empty(prefix),
        means.new_empty(prefix + (3,)),
        means.new_empty(prefix) if calc_compensations else None,
    )


def _fully_fused_projection_packed_fake(
    means: Tensor,
    covars: Tensor | None,
    quats: Tensor | None,
    scales: Tensor | None,
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    sparse_grad: bool = False,
    calc_compensations: bool = False,
    camera_model: str = "pinhole",
    opacities: Tensor | None = None,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor | None,
]:
    nnz = _dynamic_size()
    batch_dims = means.shape[:-2]
    B = math.prod(batch_dims) if batch_dims else 1
    C = viewmats.shape[-3]
    return (
        _int_output(means.device, (nnz,)),
        _int_output(means.device, (nnz,)),
        _int_output(means.device, (nnz,)),
        _int_output(means.device, (B * C + 1,)),
        _int_output(means.device, (nnz, 2)),
        means.new_empty((nnz, 2)),
        means.new_empty((nnz,)),
        means.new_empty((nnz, 3)),
        means.new_empty((nnz,)) if calc_compensations else None,
    )


def _rasterize_to_pixels_extra_fake(
    means2d: Tensor,
    conics: Tensor,
    colors: Tensor,
    opacities: Tensor,
    backgrounds: Tensor | None,
    masks: Tensor | None,
    width: int,
    height: int,
    tile_size: int,
    isect_offsets: Tensor,
    flatten_ids: Tensor,
    absgrad: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    image_dims = means2d.shape[:-2]
    channels = colors.shape[-1]
    return (
        colors.new_empty(image_dims + (height, width, channels)),
        colors.new_empty(image_dims + (height, width, 1)).float(),
        colors.new_empty(image_dims + (height, width, 1)),
        _int_output(colors.device, image_dims + (height, width)),
    )


def _fully_fused_projection_with_ut_fake(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor | None,
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    eps2d: float,
    near_plane: float,
    far_plane: float,
    radius_clip: float,
    calc_compensations: bool,
    camera_model: str,
    ut_params: Any,
    radial_coeffs: Tensor | None,
    tangential_coeffs: Tensor | None,
    thin_prism_coeffs: Tensor | None,
    ftheta_coeffs: Any,
    lidar_coeffs: Any,
    external_distortion_coeffs: Any,
    rolling_shutter: int,
    viewmats_rs: Tensor | None,
    global_z_order: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
    batch_dims = means.shape[:-2]
    N = means.shape[-2]
    C = viewmats.shape[-3]
    prefix = batch_dims + (C, N)
    return (
        _int_output(means.device, prefix + (2,)),
        means.new_empty(prefix + (2,)),
        means.new_empty(prefix),
        means.new_empty(prefix + (3,)),
        means.new_empty(prefix) if calc_compensations else None,
    )


def _rasterize_to_pixels_eval3d_extra_fake(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    colors: Tensor,
    opacities: Tensor,
    backgrounds: Tensor | None,
    masks: Tensor | None,
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    tile_size: int,
    isect_offsets: Tensor,
    flatten_ids: Tensor,
    camera_model: str,
    ut_params: Any,
    rays: Tensor | None,
    radial_coeffs: Tensor | None,
    tangential_coeffs: Tensor | None,
    thin_prism_coeffs: Tensor | None,
    ftheta_coeffs: Any,
    lidar_coeffs: Any,
    external_distortion_coeffs: Any,
    rolling_shutter: int,
    viewmats_rs: Tensor | None,
    return_sample_counts: bool,
    use_hit_distance: bool,
    return_normals: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
    batch_dims = means.shape[:-2]
    C = viewmats.shape[-3]
    channels = colors.shape[-1]
    prefix = batch_dims + (C, height, width)
    return (
        colors.new_empty(prefix + (channels,)),
        colors.new_empty(prefix + (1,)),
        _int_output(means.device, prefix),
        _int_output(means.device, prefix) if return_sample_counts else None,
        means.new_empty(prefix + (3,)) if return_normals else None,
    )


def _fully_fused_projection_2dgs_fake(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    batch_dims = means.shape[:-2]
    N = means.shape[-2]
    C = viewmats.shape[-3]
    prefix = batch_dims + (C, N)
    return (
        _int_output(means.device, prefix + (2,)),
        means.new_empty(prefix + (2,)),
        means.new_empty(prefix),
        means.new_empty(prefix + (3, 3)),
        means.new_empty(prefix + (3,)),
    )


def _fully_fused_projection_packed_2dgs_fake(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    sparse_grad: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    nnz = _dynamic_size()
    return (
        _int_output(means.device, (nnz,)),
        _int_output(means.device, (nnz,)),
        _int_output(means.device, (nnz,)),
        _int_output(means.device, (nnz, 2)),
        means.new_empty((nnz, 2)),
        means.new_empty((nnz,)),
        means.new_empty((nnz, 3, 3)),
        means.new_empty((nnz, 3)),
    )


def _rasterize_to_pixels_2dgs_extra_fake(
    means2d: Tensor,
    ray_transforms: Tensor,
    colors: Tensor,
    opacities: Tensor,
    normals: Tensor,
    densify: Tensor,
    backgrounds: Tensor | None,
    masks: Tensor | None,
    width: int,
    height: int,
    tile_size: int,
    isect_offsets: Tensor,
    flatten_ids: Tensor,
    absgrad: bool = False,
    distloss: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    image_dims = means2d.shape[:-2]
    channels = colors.shape[-1]
    prefix = image_dims + (height, width)
    return (
        colors.new_empty(prefix + (channels,)),
        colors.new_empty(prefix + (1,)).float(),
        colors.new_empty(prefix + (3,)),
        colors.new_empty(prefix + (1,)),
        colors.new_empty(prefix + (1,)),
        colors.new_empty(prefix + (1,)),
        _int_output(colors.device, prefix),
        _int_output(colors.device, prefix),
    )


def _quat_scale_to_covar_preci_setup_context(
    ctx: Any, inputs: Any, output: Any
) -> None:
    quats, scales, compute_covar, compute_preci, triu = inputs
    ctx.save_for_backward(quats, scales)
    ctx.compute_covar = compute_covar
    ctx.compute_preci = compute_preci
    ctx.triu = triu


def _quat_scale_to_covar_preci_backward(
    ctx: Any, v_covars: Tensor | None, v_precis: Tensor | None
) -> tuple[Tensor, Tensor, None, None, None]:
    quats, scales = ctx.saved_tensors
    if ctx.compute_covar and v_covars is not None and v_covars.is_sparse:
        v_covars = v_covars.to_dense()
    if ctx.compute_preci and v_precis is not None and v_precis.is_sparse:
        v_precis = v_precis.to_dense()
    v_quats, v_scales = torch.ops.gsplat.quat_scale_to_covar_preci_bwd(
        quats,
        scales,
        ctx.triu,
        v_covars.contiguous()
        if ctx.compute_covar and v_covars is not None
        else None,
        v_precis.contiguous()
        if ctx.compute_preci and v_precis is not None
        else None,
    )
    return v_quats, v_scales, None, None, None


def _spherical_harmonics_setup_context(
    ctx: Any, inputs: Any, output: Any
) -> None:
    sh_degree, dirs, coeffs, masks = inputs
    ctx.save_for_backward(dirs, coeffs, masks)
    ctx.sh_degree = sh_degree
    ctx.num_bases = coeffs.shape[-2]


def _spherical_harmonics_backward(
    ctx: Any, v_colors: Tensor
) -> tuple[None, Tensor | None, Tensor, None]:
    dirs, coeffs, masks = ctx.saved_tensors
    compute_v_dirs = ctx.needs_input_grad[1]
    v_coeffs, v_dirs = torch.ops.gsplat.spherical_harmonics_bwd(
        ctx.num_bases,
        ctx.sh_degree,
        dirs,
        coeffs,
        masks,
        v_colors.contiguous(),
        compute_v_dirs,
    )
    return None, v_dirs if compute_v_dirs else None, v_coeffs, None


def _proj_setup_context(ctx: Any, inputs: Any, output: Any) -> None:
    means, covars, Ks, width, height, camera_model = inputs
    ctx.save_for_backward(means, covars, Ks)
    ctx.width = width
    ctx.height = height
    ctx.camera_model_type = _camera_model_type(camera_model)


def _proj_backward(ctx: Any, v_means2d: Tensor, v_covars2d: Tensor) -> tuple:
    means, covars, Ks = ctx.saved_tensors
    v_means, v_covars = torch.ops.gsplat.projection_ewa_simple_bwd(
        means,
        covars,
        Ks,
        ctx.width,
        ctx.height,
        ctx.camera_model_type,
        v_means2d.contiguous(),
        v_covars2d.contiguous(),
    )
    return v_means, v_covars, None, None, None, None


def _fully_fused_projection_setup_context(
    ctx: Any, inputs: Any, output: Any
) -> None:
    (
        means,
        covars,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d,
        _near_plane,
        _far_plane,
        _radius_clip,
        _calc_compensations,
        camera_model,
        _opacities,
    ) = inputs
    radii, _means2d, _depths, conics, compensations = output
    ctx.save_for_backward(
        means, covars, quats, scales, viewmats, Ks, radii, conics, compensations
    )
    ctx.width = width
    ctx.height = height
    ctx.eps2d = eps2d
    ctx.camera_model_type = _camera_model_type(camera_model)


def _fully_fused_projection_backward(
    ctx: Any,
    _v_radii: Tensor | None,
    v_means2d: Tensor,
    v_depths: Tensor,
    v_conics: Tensor,
    v_compensations: Tensor | None,
) -> tuple:
    (
        means,
        covars,
        quats,
        scales,
        viewmats,
        Ks,
        radii,
        conics,
        compensations,
    ) = ctx.saved_tensors
    if v_compensations is not None:
        v_compensations = v_compensations.contiguous()
    v_means, v_covars, v_quats, v_scales, v_viewmats = (
        torch.ops.gsplat.projection_ewa_3dgs_fused_bwd(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            ctx.width,
            ctx.height,
            ctx.eps2d,
            ctx.camera_model_type,
            radii,
            conics,
            compensations,
            v_means2d.contiguous(),
            v_depths.contiguous(),
            v_conics.contiguous(),
            v_compensations,
            ctx.needs_input_grad[4],
        )
    )
    if not ctx.needs_input_grad[0]:
        v_means = None
    if not ctx.needs_input_grad[1]:
        v_covars = None
    if not ctx.needs_input_grad[2]:
        v_quats = None
    if not ctx.needs_input_grad[3]:
        v_scales = None
    if not ctx.needs_input_grad[4]:
        v_viewmats = None
    return (
        v_means,
        v_covars,
        v_quats,
        v_scales,
        v_viewmats,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


def _fully_fused_projection_packed_setup_context(
    ctx: Any, inputs: Any, output: Any
) -> None:
    (
        means,
        covars,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d,
        _near_plane,
        _far_plane,
        _radius_clip,
        sparse_grad,
        _calc_compensations,
        camera_model,
        _opacities,
    ) = inputs
    (
        batch_ids,
        camera_ids,
        gaussian_ids,
        _indptr,
        _radii,
        _means2d,
        _depths,
        conics,
        compensations,
    ) = output
    ctx.save_for_backward(
        batch_ids,
        camera_ids,
        gaussian_ids,
        means,
        covars,
        quats,
        scales,
        viewmats,
        Ks,
        conics,
        compensations,
    )
    ctx.width = width
    ctx.height = height
    ctx.eps2d = eps2d
    ctx.sparse_grad = sparse_grad
    ctx.camera_model_type = _camera_model_type(camera_model)


def _fully_fused_projection_packed_backward(
    ctx: Any,
    _v_batch_ids: Tensor | None,
    _v_camera_ids: Tensor | None,
    _v_gaussian_ids: Tensor | None,
    _v_indptr: Tensor | None,
    _v_radii: Tensor | None,
    v_means2d: Tensor,
    v_depths: Tensor,
    v_conics: Tensor,
    v_compensations: Tensor | None,
) -> tuple:
    (
        batch_ids,
        camera_ids,
        gaussian_ids,
        means,
        covars,
        quats,
        scales,
        viewmats,
        Ks,
        conics,
        compensations,
    ) = ctx.saved_tensors
    if v_compensations is not None:
        v_compensations = v_compensations.contiguous()
    v_means, v_covars, v_quats, v_scales, v_viewmats = (
        torch.ops.gsplat.projection_ewa_3dgs_packed_bwd(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            ctx.width,
            ctx.height,
            ctx.eps2d,
            ctx.camera_model_type,
            batch_ids,
            camera_ids,
            gaussian_ids,
            conics,
            compensations,
            v_means2d.contiguous(),
            v_depths.contiguous(),
            v_conics.contiguous(),
            v_compensations,
            ctx.needs_input_grad[4],
            ctx.sparse_grad,
        )
    )
    if not ctx.needs_input_grad[0]:
        v_means = None
    elif ctx.sparse_grad:
        v_means = torch.sparse_coo_tensor(
            indices=gaussian_ids[None],
            values=v_means,
            size=means.shape,
            is_coalesced=len(viewmats) == 1,
        )
    if not ctx.needs_input_grad[1]:
        v_covars = None
    elif ctx.sparse_grad:
        v_covars = torch.sparse_coo_tensor(
            indices=gaussian_ids[None],
            values=v_covars,
            size=covars.shape,
            is_coalesced=len(viewmats) == 1,
        )
    if not ctx.needs_input_grad[2]:
        v_quats = None
    elif ctx.sparse_grad:
        v_quats = torch.sparse_coo_tensor(
            indices=gaussian_ids[None],
            values=v_quats,
            size=quats.shape,
            is_coalesced=len(viewmats) == 1,
        )
    if not ctx.needs_input_grad[3]:
        v_scales = None
    elif ctx.sparse_grad:
        v_scales = torch.sparse_coo_tensor(
            indices=gaussian_ids[None],
            values=v_scales,
            size=scales.shape,
            is_coalesced=len(viewmats) == 1,
        )
    if not ctx.needs_input_grad[4]:
        v_viewmats = None
    return (
        v_means,
        v_covars,
        v_quats,
        v_scales,
        v_viewmats,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


def _rasterize_to_pixels_extra_setup_context(
    ctx: Any, inputs: Any, output: Any
) -> None:
    (
        means2d,
        conics,
        colors,
        opacities,
        backgrounds,
        masks,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        absgrad,
    ) = inputs
    _render_colors, _render_alphas, raw_render_alphas, last_ids = output
    ctx.save_for_backward(
        means2d,
        conics,
        colors,
        opacities,
        backgrounds,
        masks,
        isect_offsets,
        flatten_ids,
        raw_render_alphas,
        last_ids,
    )
    ctx.width = width
    ctx.height = height
    ctx.tile_size = tile_size
    ctx.absgrad = absgrad


def _rasterize_to_pixels_extra_backward(
    ctx: Any,
    v_render_colors: Tensor,
    v_render_alphas: Tensor,
    _v_raw_render_alphas: Tensor | None,
    _v_last_ids: Tensor | None,
) -> tuple:
    (
        means2d,
        conics,
        colors,
        opacities,
        backgrounds,
        masks,
        isect_offsets,
        flatten_ids,
        raw_render_alphas,
        last_ids,
    ) = ctx.saved_tensors
    v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities = (
        torch.ops.gsplat.rasterize_to_pixels_3dgs_bwd(
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            ctx.width,
            ctx.height,
            ctx.tile_size,
            isect_offsets,
            flatten_ids,
            raw_render_alphas,
            last_ids,
            v_render_colors.contiguous(),
            v_render_alphas.contiguous(),
            ctx.absgrad,
        )
    )
    if ctx.absgrad:
        means2d.absgrad = v_means2d_abs
    v_backgrounds = None
    if ctx.needs_input_grad[4]:
        v_backgrounds = (
            v_render_colors * (1.0 - raw_render_alphas).float()
        ).sum(dim=(-3, -2))
    return (
        v_means2d,
        v_conics,
        v_colors,
        v_opacities,
        v_backgrounds,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


def _rasterize_to_pixels_eval3d_extra_setup_context(
    ctx: Any, inputs: Any, output: Any
) -> None:
    (
        means,
        quats,
        scales,
        colors,
        opacities,
        backgrounds,
        masks,
        viewmats,
        Ks,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        camera_model,
        ut_params,
        rays,
        radial_coeffs,
        tangential_coeffs,
        thin_prism_coeffs,
        ftheta_coeffs,
        lidar_coeffs,
        external_distortion_coeffs,
        rolling_shutter,
        viewmats_rs,
        _return_sample_counts,
        use_hit_distance,
        _return_normals,
    ) = inputs
    _render_colors, render_alphas, last_ids, _sample_counts, _render_normals = (
        output
    )
    ctx.save_for_backward(
        means,
        quats,
        scales,
        colors,
        opacities,
        backgrounds,
        masks,
        viewmats,
        viewmats_rs,
        Ks,
        rays,
        radial_coeffs,
        tangential_coeffs,
        thin_prism_coeffs,
        isect_offsets,
        flatten_ids,
        render_alphas,
        last_ids,
    )
    ctx.width = width
    ctx.height = height
    ctx.ut_params = ut_params
    ctx.rs_type = rolling_shutter
    ctx.camera_model_type = _camera_model_type(camera_model)
    ctx.tile_size = tile_size
    ctx.ftheta_coeffs = ftheta_coeffs
    ctx.lidar_coeffs = lidar_coeffs
    ctx.external_distortion_coeffs = external_distortion_coeffs
    ctx.use_hit_distance = use_hit_distance


def _rasterize_to_pixels_eval3d_extra_backward(
    ctx: Any,
    v_render_colors: Tensor,
    v_render_alphas: Tensor,
    _v_last_ids: Tensor | None,
    _v_sample_counts: Tensor | None,
    v_render_normals: Tensor | None,
) -> tuple:
    (
        means,
        quats,
        scales,
        colors,
        opacities,
        backgrounds,
        masks,
        viewmats,
        viewmats_rs,
        Ks,
        rays,
        radial_coeffs,
        tangential_coeffs,
        thin_prism_coeffs,
        isect_offsets,
        flatten_ids,
        render_alphas,
        last_ids,
    ) = ctx.saved_tensors
    v_means, v_quats, v_scales, v_colors, v_opacities, v_rays = (
        torch.ops.gsplat.rasterize_to_pixels_from_world_3dgs_bwd(
            means,
            quats,
            scales,
            colors,
            opacities,
            backgrounds,
            masks,
            ctx.width,
            ctx.height,
            ctx.tile_size,
            viewmats,
            viewmats_rs,
            Ks,
            ctx.camera_model_type,
            ctx.ut_params,
            ctx.rs_type,
            rays,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs,
            ctx.ftheta_coeffs,
            ctx.lidar_coeffs,
            ctx.external_distortion_coeffs,
            isect_offsets,
            flatten_ids,
            ctx.use_hit_distance,
            render_alphas,
            last_ids,
            v_render_colors.contiguous(),
            v_render_alphas.contiguous(),
            v_render_normals.contiguous()
            if v_render_normals is not None
            else None,
        )
    )
    v_backgrounds = None
    if ctx.needs_input_grad[5]:
        v_backgrounds = (v_render_colors * (1.0 - render_alphas).float()).sum(
            dim=(-3, -2)
        )
    return (
        v_means,
        v_quats,
        v_scales,
        v_colors,
        v_opacities,
        v_backgrounds,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        v_rays,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


def _fully_fused_projection_2dgs_setup_context(
    ctx: Any, inputs: Any, output: Any
) -> None:
    means, quats, scales, viewmats, Ks, width, height, eps2d, *_ = inputs
    radii, _means2d, _depths, ray_transforms, normals = output
    ctx.save_for_backward(
        means, quats, scales, viewmats, Ks, radii, ray_transforms, normals
    )
    ctx.width = width
    ctx.height = height
    ctx.eps2d = eps2d


def _fully_fused_projection_2dgs_backward(
    ctx: Any,
    _v_radii: Tensor | None,
    v_means2d: Tensor,
    v_depths: Tensor,
    v_ray_transforms: Tensor,
    v_normals: Tensor,
) -> tuple:
    means, quats, scales, viewmats, Ks, radii, ray_transforms, _normals = (
        ctx.saved_tensors
    )
    v_means, v_quats, v_scales, v_viewmats = (
        torch.ops.gsplat.projection_2dgs_fused_bwd(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            ctx.width,
            ctx.height,
            radii,
            ray_transforms,
            v_means2d.contiguous(),
            v_depths.contiguous(),
            v_normals.contiguous(),
            v_ray_transforms.contiguous(),
            ctx.needs_input_grad[3],
        )
    )
    if not ctx.needs_input_grad[0]:
        v_means = None
    if not ctx.needs_input_grad[1]:
        v_quats = None
    if not ctx.needs_input_grad[2]:
        v_scales = None
    if not ctx.needs_input_grad[3]:
        v_viewmats = None
    return (
        v_means,
        v_quats,
        v_scales,
        v_viewmats,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


def _fully_fused_projection_packed_2dgs_setup_context(
    ctx: Any, inputs: Any, output: Any
) -> None:
    means, quats, scales, viewmats, Ks, width, height, *_rest, sparse_grad = (
        inputs
    )
    (
        batch_ids,
        camera_ids,
        gaussian_ids,
        _radii,
        _means2d,
        _depths,
        ray_transforms,
        _normals,
    ) = output
    ctx.save_for_backward(
        batch_ids,
        camera_ids,
        gaussian_ids,
        means,
        quats,
        scales,
        viewmats,
        Ks,
        ray_transforms,
    )
    ctx.width = width
    ctx.height = height
    ctx.sparse_grad = sparse_grad


def _fully_fused_projection_packed_2dgs_backward(
    ctx: Any,
    _v_batch_ids: Tensor | None,
    _v_camera_ids: Tensor | None,
    _v_gaussian_ids: Tensor | None,
    _v_radii: Tensor | None,
    v_means2d: Tensor,
    v_depths: Tensor,
    v_ray_transforms: Tensor,
    v_normals: Tensor,
) -> tuple:
    (
        batch_ids,
        camera_ids,
        gaussian_ids,
        means,
        quats,
        scales,
        viewmats,
        Ks,
        ray_transforms,
    ) = ctx.saved_tensors
    v_means, v_quats, v_scales, v_viewmats = (
        torch.ops.gsplat.projection_2dgs_packed_bwd(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            ctx.width,
            ctx.height,
            batch_ids,
            camera_ids,
            gaussian_ids,
            ray_transforms,
            v_means2d.contiguous(),
            v_depths.contiguous(),
            v_ray_transforms.contiguous(),
            v_normals.contiguous(),
            ctx.needs_input_grad[3],
            ctx.sparse_grad,
        )
    )
    if not ctx.needs_input_grad[0]:
        v_means = None
    elif ctx.sparse_grad:
        v_means = torch.sparse_coo_tensor(
            indices=gaussian_ids[None],
            values=v_means,
            size=means.shape,
            is_coalesced=len(viewmats) == 1,
        )
    if not ctx.needs_input_grad[1]:
        v_quats = None
    elif ctx.sparse_grad:
        v_quats = torch.sparse_coo_tensor(
            indices=gaussian_ids[None],
            values=v_quats,
            size=quats.shape,
            is_coalesced=len(viewmats) == 1,
        )
    if not ctx.needs_input_grad[2]:
        v_scales = None
    elif ctx.sparse_grad:
        v_scales = torch.sparse_coo_tensor(
            indices=gaussian_ids[None],
            values=v_scales,
            size=scales.shape,
            is_coalesced=len(viewmats) == 1,
        )
    if not ctx.needs_input_grad[3]:
        v_viewmats = None
    return (
        v_means,
        v_quats,
        v_scales,
        v_viewmats,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


def _rasterize_to_pixels_2dgs_extra_setup_context(
    ctx: Any, inputs: Any, output: Any
) -> None:
    (
        means2d,
        ray_transforms,
        colors,
        opacities,
        normals,
        densify,
        backgrounds,
        masks,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        absgrad,
        distloss,
    ) = inputs
    (
        render_colors,
        _render_alphas,
        _render_normals,
        _render_distort,
        _render_median,
        raw_render_alphas,
        last_ids,
        median_ids,
    ) = output
    ctx.save_for_backward(
        means2d,
        ray_transforms,
        colors,
        opacities,
        normals,
        densify,
        backgrounds,
        masks,
        isect_offsets,
        flatten_ids,
        render_colors,
        raw_render_alphas,
        last_ids,
        median_ids,
    )
    ctx.width = width
    ctx.height = height
    ctx.tile_size = tile_size
    ctx.absgrad = absgrad
    ctx.distloss = distloss


def _rasterize_to_pixels_2dgs_extra_backward(
    ctx: Any,
    v_render_colors: Tensor,
    v_render_alphas: Tensor,
    v_render_normals: Tensor,
    v_render_distort: Tensor,
    v_render_median: Tensor,
    _v_raw_render_alphas: Tensor | None,
    _v_last_ids: Tensor | None,
    _v_median_ids: Tensor | None,
) -> tuple:
    (
        means2d,
        ray_transforms,
        colors,
        opacities,
        normals,
        densify,
        backgrounds,
        masks,
        isect_offsets,
        flatten_ids,
        render_colors,
        raw_render_alphas,
        last_ids,
        median_ids,
    ) = ctx.saved_tensors
    (
        v_means2d_abs,
        v_means2d,
        v_ray_transforms,
        v_colors,
        v_opacities,
        v_normals,
        v_densify,
    ) = torch.ops.gsplat.rasterize_to_pixels_2dgs_bwd(
        means2d,
        ray_transforms,
        colors,
        opacities,
        normals,
        densify,
        backgrounds,
        masks,
        ctx.width,
        ctx.height,
        ctx.tile_size,
        isect_offsets,
        flatten_ids,
        render_colors,
        raw_render_alphas,
        last_ids,
        median_ids,
        v_render_colors.contiguous(),
        v_render_alphas.contiguous(),
        v_render_normals.contiguous(),
        v_render_distort.contiguous(),
        v_render_median.contiguous(),
        ctx.absgrad,
    )
    if ctx.absgrad:
        means2d.absgrad = v_means2d_abs
    v_backgrounds = None
    if ctx.needs_input_grad[6]:
        v_backgrounds = (
            v_render_colors * (1.0 - raw_render_alphas).float()
        ).sum(dim=(-3, -2))
    return (
        v_means2d,
        v_ray_transforms,
        v_colors,
        v_opacities,
        v_normals,
        v_densify,
        v_backgrounds,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


quat_scale_to_covar_preci.register_fake(_quat_scale_to_covar_preci_fake)
spherical_harmonics.register_fake(_spherical_harmonics_fake)
proj.register_fake(_proj_fake)
fully_fused_projection.register_fake(_fully_fused_projection_fake)
fully_fused_projection_packed.register_fake(_fully_fused_projection_packed_fake)
rasterize_to_pixels_extra.register_fake(_rasterize_to_pixels_extra_fake)
fully_fused_projection_2dgs.register_fake(_fully_fused_projection_2dgs_fake)
fully_fused_projection_packed_2dgs.register_fake(
    _fully_fused_projection_packed_2dgs_fake
)
rasterize_to_pixels_2dgs_extra.register_fake(
    _rasterize_to_pixels_2dgs_extra_fake
)

quat_scale_to_covar_preci.register_autograd(
    _quat_scale_to_covar_preci_backward,
    setup_context=_quat_scale_to_covar_preci_setup_context,
)
spherical_harmonics.register_autograd(
    _spherical_harmonics_backward,
    setup_context=_spherical_harmonics_setup_context,
)
proj.register_autograd(_proj_backward, setup_context=_proj_setup_context)
fully_fused_projection.register_autograd(
    _fully_fused_projection_backward,
    setup_context=_fully_fused_projection_setup_context,
)
fully_fused_projection_packed.register_autograd(
    _fully_fused_projection_packed_backward,
    setup_context=_fully_fused_projection_packed_setup_context,
)
rasterize_to_pixels_extra.register_autograd(
    _rasterize_to_pixels_extra_backward,
    setup_context=_rasterize_to_pixels_extra_setup_context,
)
fully_fused_projection_2dgs.register_autograd(
    _fully_fused_projection_2dgs_backward,
    setup_context=_fully_fused_projection_2dgs_setup_context,
)
fully_fused_projection_packed_2dgs.register_autograd(
    _fully_fused_projection_packed_2dgs_backward,
    setup_context=_fully_fused_projection_packed_2dgs_setup_context,
)
rasterize_to_pixels_2dgs_extra.register_autograd(
    _rasterize_to_pixels_2dgs_extra_backward,
    setup_context=_rasterize_to_pixels_2dgs_extra_setup_context,
)

if _has_custom_classes():
    fully_fused_projection_with_ut.register_fake(
        _fully_fused_projection_with_ut_fake
    )
    rasterize_to_pixels_eval3d_extra.register_fake(
        _rasterize_to_pixels_eval3d_extra_fake
    )
    rasterize_to_pixels_eval3d_extra.register_autograd(
        _rasterize_to_pixels_eval3d_extra_backward,
        setup_context=_rasterize_to_pixels_eval3d_extra_setup_context,
    )
