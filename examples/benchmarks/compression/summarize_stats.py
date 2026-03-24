# SPDX-FileCopyrightText: Copyright 2023-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""Script to summarize compression benchmark statistics."""

import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import tyro


def main(results_dir: str, scenes: list[str], stage: str = "compress") -> None:
    """Summarize compression benchmark statistics for given scenes."""
    print("scenes:", scenes)

    results_path = Path(results_dir)
    summary = defaultdict(list)
    for scene in scenes:
        scene_dir = results_path / scene

        if stage == "compress":
            zip_path = scene_dir / "compression.zip"
            if zip_path.exists():
                zip_path.unlink()
            subprocess.run(
                f"zip -r {zip_path} {scene_dir / 'compression'}/", shell=True
            )
            out = subprocess.run(
                f"stat -c%s {zip_path}", shell=True, capture_output=True
            )
            size = int(out.stdout)
            summary["size"].append(size)

        with (scene_dir / "stats" / f"{stage}_step29999.json").open() as f:
            stats = json.load(f)
            for k, v in stats.items():
                summary[k].append(v)

    for k, v in summary.items():
        summary[k] = np.mean(v)
    summary["scenes"] = scenes

    with (results_path / f"{stage}_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    tyro.cli(main)
