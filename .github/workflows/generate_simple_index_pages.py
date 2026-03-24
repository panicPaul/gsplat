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

"""Generate PEP 503-style index pages for released Python wheels."""

import argparse
import re
from pathlib import Path
from typing import TypedDict

import requests
from jinja2 import Template


class WheelInfo(TypedDict):
    """Metadata for a released Python wheel."""

    release_name: str
    wheel_name: str
    download_url: str
    package_name: str
    local_version: str | None


def _github_repo() -> str:
    """Return the GitHub repository in `owner/repo` form."""
    import os

    repo = os.getenv("GITHUB_REPOSITORY")
    if repo is None:
        raise RuntimeError("GITHUB_REPOSITORY is not set")
    return repo


def list_python_wheels() -> list[WheelInfo]:
    """Fetch all released wheel assets from the GitHub releases API."""
    releases_url = f"https://api.github.com/repos/{_github_repo()}/releases"

    response = requests.get(releases_url)

    if response.status_code != 200:
        raise Exception(
            f"Failed to fetch releases: {response.status_code} {response.text}"
        )

    releases = response.json()

    wheel_files: list[WheelInfo] = []

    # Iterate through releases and assets
    for release in releases:
        assets = release.get("assets", [])
        for asset in assets:
            filename = asset["name"]
            if filename.endswith(".whl"):
                pattern = r"^(?P<name>[\w\d_.]+)-(?P<version>[\d.]+)(?P<local>\+[\w\d.]+)?-(?P<python_tag>[\w]+)-(?P<abi_tag>[\w]+)-(?P<platform_tag>[\w]+)\.whl"

                match = re.match(pattern, filename)

                if match:
                    local_version = match.group("local")
                    if local_version:
                        local_version = local_version.lstrip(
                            "+"
                        )  # Return the local version without the '+' sign
                    else:
                        local_version = None
                else:
                    raise ValueError(f"Invalid wheel filename: {filename}")
                wheel_files.append(
                    {
                        "release_name": release["name"],
                        "wheel_name": asset["name"],
                        "download_url": asset["browser_download_url"],
                        "package_name": match.group("name"),
                        "local_version": local_version,
                    }
                )

    return wheel_files


def generate_simple_index_htmls(wheels: list[WheelInfo], outdir: Path) -> None:
    """Write simple index pages for all wheels and package groups."""
    template_versions_str = """
    <!DOCTYPE html>
    <html>
    <head><title>Python wheels links for {{ repo_name }}</title></head>
    <body>
    <h1>Python wheels for {{ repo_name }}</h1>

    {% for wheel in wheels %}
    <a href="{{ wheel.download_url }}">{{ wheel.wheel_name }}</a><br/>
    {% endfor %}

    </body>
    </html>
    """

    template_packages_str = """
    <html>
    <body>
    {% for package_name in package_names %}
        <a href="{{package_name}}/">{{package_name}}</a><br/>
    {% endfor %}
    </body>
    </html>
    """

    # Create a Jinja2 Template object from the string
    template_versions = Template(template_versions_str)
    template_packages = Template(template_packages_str)

    # group the wheels by package name
    packages: dict[str, list[WheelInfo]] = {}
    for wheel in wheels:
        package_name = wheel["package_name"]
        if package_name not in packages:
            packages[package_name] = []
        packages[package_name].append(wheel)

    # Render the HTML the list the package names
    html_content = template_packages.render(
        package_names=[str(k) for k in packages]
    )
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "index.html").open("w") as file:
        file.write(html_content)

    # for each package, render the HTML to list the wheels
    for package_name, wheels in packages.items():
        html_page = template_versions.render(
            repo_name=_github_repo(), wheels=wheels
        )
        package_dir = outdir / package_name
        package_dir.mkdir(parents=True, exist_ok=True)
        with (package_dir / "index.html").open("w") as file:
            file.write(html_page)


def generate_all_pages(outdir: Path) -> None:
    """Generate the root index and per-local-version sub-indexes."""
    wheels = list_python_wheels()
    if wheels:
        print("Python Wheels found in releases:")
        for wheel in wheels:
            print(
                f"Release: {wheel['release_name']}, Wheel: {wheel['wheel_name']}, URL: {wheel['download_url']}"
            )
    else:
        print("No Python wheels found in the releases.")

    # Generate Simple Index HTML pages the wheel with all local versions
    generate_simple_index_htmls(wheels, outdir=outdir)

    # group wheels per local version
    wheels_per_local_version: dict[str, list[WheelInfo]] = {}
    for wheel in wheels:
        local_version = wheel["local_version"]
        if local_version is None:
            continue
        if local_version not in wheels_per_local_version:
            wheels_per_local_version[local_version] = []
        wheels_per_local_version[local_version].append(wheel)

    # create a subdirectory for each local version
    for local_version, wheels in wheels_per_local_version.items():
        version_outdir = outdir / local_version
        version_outdir.mkdir(parents=True, exist_ok=True)
        generate_simple_index_htmls(wheels, outdir=version_outdir)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Generate Python Wheels Index Pages"
    )
    argparser.add_argument(
        "--outdir", help="Output directory for the index pages", default="."
    )
    args = argparser.parse_args()
    generate_all_pages(Path(args.outdir))
