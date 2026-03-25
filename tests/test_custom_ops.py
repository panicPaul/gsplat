# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gsplat
import pytest
import torch
from gsplat.cuda._backend import _C
from gsplat.cuda._wrapper import (
    FThetaCameraDistortionParameters,
    UnscentedTransformParameters,
)
from torch._subclasses.fake_tensor import FakeTensorMode

if _C is None:
    pytest.skip("gsplat CUDA extension not available", allow_module_level=True)


device = torch.device("cuda:0")


def _camera_intrinsics(width: int, height: int, count: int = 1) -> torch.Tensor:
    fx, fy, cx, cy = width, width, width / 2.0, height / 2.0
    K = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        device=device,
        dtype=torch.float32,
    )
    return K.expand(count, 3, 3).contiguous()


def _viewmats(count: int = 1) -> torch.Tensor:
    return (
        torch.eye(4, device=device, dtype=torch.float32)
        .expand(count, 4, 4)
        .contiguous()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_registered_custom_ops_available():
    assert hasattr(torch.ops.gsplat, "quat_scale_to_covar_preci")
    assert hasattr(torch.ops.gsplat, "proj")
    assert hasattr(torch.ops.gsplat, "fully_fused_projection")
    assert hasattr(torch.ops.gsplat, "fully_fused_projection_packed")
    assert hasattr(torch.ops.gsplat, "rasterize_to_pixels_extra")
    assert hasattr(torch.ops.gsplat, "fully_fused_projection_2dgs")
    assert hasattr(torch.ops.gsplat, "fully_fused_projection_packed_2dgs")
    assert hasattr(torch.ops.gsplat, "rasterize_to_pixels_2dgs_extra")
    assert hasattr(torch.ops.gsplat, "fully_fused_projection_with_ut")
    assert hasattr(torch.ops.gsplat, "rasterize_to_pixels_eval3d_extra")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_opcheck_core_3dgs_ops():
    quats = torch.randn(4, 4, device=device, dtype=torch.float32)
    scales = torch.rand(4, 3, device=device, dtype=torch.float32) + 0.1

    torch.library.opcheck(
        torch.ops.gsplat.quat_scale_to_covar_preci.default,
        (quats, scales, True, True, False),
    )

    means = torch.randn(1, 1, 4, 3, device=device, dtype=torch.float32)
    covars = (
        torch.eye(3, device=device, dtype=torch.float32)
        .expand(1, 1, 4, 3, 3)
        .contiguous()
    )
    Ks = _camera_intrinsics(32, 24).view(1, 1, 3, 3)

    torch.library.opcheck(
        torch.ops.gsplat.proj.default,
        (means, covars, Ks, 32, 24, "pinhole"),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_opcheck_core_2dgs_ops():
    N = 8
    means = torch.randn(N, 3, device=device, dtype=torch.float32)
    quats = torch.nn.functional.normalize(
        torch.randn(N, 4, device=device, dtype=torch.float32), dim=-1
    )
    scales = torch.ones(N, 3, device=device, dtype=torch.float32) * 0.1
    viewmats = _viewmats(1)
    Ks = _camera_intrinsics(32, 24)

    torch.library.opcheck(
        torch.ops.gsplat.fully_fused_projection_2dgs.default,
        (means, quats, scales, viewmats, Ks, 32, 24, 0.3, 0.01, 1e10, 0.0),
        test_utils=(
            "test_schema",
            "test_autograd_registration",
            "test_faketensor",
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_fake_tensor_support_for_registered_paths():
    means = torch.randn(6, 3, device=device, dtype=torch.float32)
    quats = torch.nn.functional.normalize(
        torch.randn(6, 4, device=device, dtype=torch.float32), dim=-1
    )
    scales = torch.rand(6, 3, device=device, dtype=torch.float32) + 0.1
    opacities = torch.rand(6, device=device, dtype=torch.float32)
    covars = torch.rand(6, 6, device=device, dtype=torch.float32)
    viewmats = _viewmats(2)
    Ks = _camera_intrinsics(32, 24, count=2)

    with FakeTensorMode(allow_non_fake_inputs=True) as mode:
        fake_means = mode.from_tensor(means)
        fake_quats = mode.from_tensor(quats)
        fake_scales = mode.from_tensor(scales)
        fake_opacities = mode.from_tensor(opacities)
        fake_covars = mode.from_tensor(covars)
        fake_viewmats = mode.from_tensor(viewmats)
        fake_Ks = mode.from_tensor(Ks)

        projected = torch.ops.gsplat.fully_fused_projection(
            fake_means,
            fake_covars,
            None,
            None,
            fake_viewmats,
            fake_Ks,
            32,
            24,
            0.3,
            0.01,
            1e10,
            0.0,
            False,
            "pinhole",
            fake_opacities,
        )
        assert projected[0].shape[-1] == 2

        projected_2dgs = torch.ops.gsplat.fully_fused_projection_2dgs(
            fake_means,
            fake_quats,
            fake_scales,
            fake_viewmats,
            fake_Ks,
            32,
            24,
            0.3,
            0.01,
            1e10,
            0.0,
        )
        assert projected_2dgs[0].shape[-1] == 2

        ut = UnscentedTransformParameters()
        ftheta = FThetaCameraDistortionParameters()
        ut_out = torch.ops.gsplat.fully_fused_projection_with_ut(
            fake_means,
            fake_quats,
            fake_scales,
            fake_opacities,
            fake_viewmats,
            fake_Ks,
            32,
            24,
            0.3,
            0.01,
            1e10,
            0.0,
            False,
            "pinhole",
            ut,
            None,
            None,
            None,
            ftheta,
            None,
            None,
            4,
            None,
            True,
        )
        assert ut_out[0].shape[-1] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_torch_compile_smoke_proj():
    means = torch.randn(1, 1, 4, 3, device=device, dtype=torch.float32)
    covars = (
        torch.eye(3, device=device, dtype=torch.float32)
        .expand(1, 1, 4, 3, 3)
        .contiguous()
    )
    Ks = _camera_intrinsics(32, 24).view(1, 1, 3, 3)

    compiled = torch.compile(
        lambda a, b, c: gsplat.proj(a, b, c, 32, 24, "pinhole")[0],
        backend="eager",
    )
    out = compiled(means, covars, Ks)
    assert out.shape == (1, 1, 4, 2)
