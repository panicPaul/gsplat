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

"""Build configuration for the gsplat package."""

import os
from pathlib import Path

from setuptools import setup

BUILD_NO_CUDA = os.getenv("BUILD_NO_CUDA", "0") == "1"


def get_ext() -> type:
    """Return the custom build extension class."""
    from torch.utils.cpp_extension import BuildExtension

    ext = BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=True)

    # Use a persistent build directory so ninja can cache object files across
    # installs (e.g. pixi install after adding a dependency).
    class PersistentBuildExt(ext):
        def initialize_options(self) -> None:
            super().initialize_options()
            self.build_temp = str(
                Path(__file__).resolve().parent / "build" / "temp"
            )

    return PersistentBuildExt


def get_extensions() -> list[object]:
    """Return the CUDA extension modules for package builds."""
    # Use the same build parameters as the JIT build. However, directly
    # importing the gsplat.cuda.build module would trigger a circular
    # dependency where gsplat is imported before it is built. To avoid
    # this, we sidestep the traditional Python import mechanism and construct
    # the module directly from build.py.
    import importlib.util

    from torch.utils.cpp_extension import CUDAExtension

    spec = importlib.util.spec_from_file_location(
        "gsplat_cuda_build", Path("gsplat") / "cuda" / "build.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    params = module.get_build_parameters()

    setup_dir = Path(__file__).resolve().parent
    sources = [str(Path(s).relative_to(setup_dir)) for s in params.sources]

    extension = CUDAExtension(
        "gsplat.csrc",
        sources=sources,
        include_dirs=params.extra_include_paths,
        extra_compile_args={
            "cxx": params.extra_cflags,
            "nvcc": params.extra_cuda_cflags,
        },
        extra_link_args=params.extra_ldflags,
    )
    return [extension]


setup(
    ext_modules=get_extensions() if not BUILD_NO_CUDA else [],
    cmdclass={"build_ext": get_ext()} if not BUILD_NO_CUDA else {},
)
