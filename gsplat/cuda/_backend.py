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

"""CUDA backend loader for gsplat.

This module intentionally does not JIT-compile the extension. The gsplat CUDA
backend must be built explicitly, for example with:

    pixi run python setup.py build_ext --inplace
"""

import os
from pathlib import Path

_IMPORT_ERROR_MESSAGE = """
gsplat CUDA extension is not available.

Build the extension explicitly before importing CUDA-backed functionality:

    pixi run python setup.py build_ext --inplace

If you need camera/custom-class support, build with:

    BUILD_CAMERA_WRAPPERS=1 pixi run python setup.py build_ext --inplace
""".strip()


def _ensure_cuda_include_env() -> None:
    """Expose Pixi/conda CUDA headers to third-party JIT extensions."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return

    include_dir = Path(conda_prefix) / "targets" / "x86_64-linux" / "include"
    if not include_dir.exists():
        return

    include_dir_str = str(include_dir)
    for env_name in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"):
        current = os.environ.get(env_name)
        if not current:
            os.environ[env_name] = include_dir_str
            continue
        paths = current.split(os.pathsep)
        if include_dir_str not in paths:
            os.environ[env_name] = os.pathsep.join([include_dir_str, *paths])


_ensure_cuda_include_env()

try:
    from gsplat import csrc as _C
except (
    ImportError
) as e:  # pragma: no cover - exercised via import failure paths
    raise ImportError(_IMPORT_ERROR_MESSAGE) from e

__all__ = ["_C"]
