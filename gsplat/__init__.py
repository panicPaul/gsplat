# SPDX-FileCopyrightText: Copyright 2023-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""gsplat: A library for Gaussian splatting rendering and optimization."""

from .color_correct import color_correct_affine, color_correct_quadratic
from .compression import PngCompression
from .cuda._lidar import (
    SpinningDirection,
)
from .cuda._lidar import (
    compute_angles_to_columns_map as compute_lidar_angles_to_columns_map,
)
from .cuda._lidar import (
    compute_tiling as compute_lidar_tiling,
)
from .cuda._torch_impl import accumulate
from .cuda._torch_impl_2dgs import accumulate_2dgs
from .cuda._wrapper import (
    CameraModel,
    ExternalDistortionModelMeta,
    RollingShutterType,
    RowOffsetStructuredSpinningLidarModelParameters,
    RowOffsetStructuredSpinningLidarModelParametersExt,
    fully_fused_projection,
    fully_fused_projection_2dgs,
    fully_fused_projection_with_ut,
    has_2dgs,
    has_3dgs,
    has_3dgut,
    has_adam,
    has_reloc,
    isect_offset_encode,
    isect_tiles,
    isect_tiles_lidar,
    proj,
    quat_scale_to_covar_preci,
    rasterize_to_indices_in_range,
    rasterize_to_indices_in_range_2dgs,
    rasterize_to_pixels,
    rasterize_to_pixels_2dgs,
    rasterize_to_pixels_eval3d,
    spherical_harmonics,
    world_to_cam,
)
from .exporter import export_splats
from .optimizers import SelectiveAdam
from .rendering import (
    RasterizeMode,
    RenderMode,
    rasterization,
    rasterization_2dgs,
    rasterization_2dgs_inria_wrapper,
    rasterization_inria_wrapper,
)
from .strategy import DefaultStrategy, MCMCStrategy, Strategy
from .version import __version__

__all__ = [
    "CameraModel",
    "DefaultStrategy",
    "ExternalDistortionModelMeta",
    "MCMCStrategy",
    "PngCompression",
    "RasterizeMode",
    "RenderMode",
    "RollingShutterType",
    "RowOffsetStructuredSpinningLidarModelParameters",
    "RowOffsetStructuredSpinningLidarModelParametersExt",
    "SelectiveAdam",
    "SpinningDirection",
    "Strategy",
    "__version__",
    "accumulate",
    "accumulate_2dgs",
    "color_correct_affine",
    "color_correct_quadratic",
    "compute_lidar_angles_to_columns_map",
    "compute_lidar_tiling",
    "export_splats",
    "fully_fused_projection",
    "fully_fused_projection_2dgs",
    "fully_fused_projection_with_ut",
    "has_2dgs",
    "has_3dgs",
    "has_3dgut",
    "has_adam",
    "has_reloc",
    "isect_offset_encode",
    "isect_tiles",
    "isect_tiles_lidar",
    "proj",
    "quat_scale_to_covar_preci",
    "rasterization",
    "rasterization_2dgs",
    "rasterization_2dgs_inria_wrapper",
    "rasterization_inria_wrapper",
    "rasterize_to_indices_in_range",
    "rasterize_to_indices_in_range_2dgs",
    "rasterize_to_pixels",
    "rasterize_to_pixels_2dgs",
    "rasterize_to_pixels_eval3d",
    "spherical_harmonics",
    "world_to_cam",
]
