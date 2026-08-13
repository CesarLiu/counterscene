#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.
#  Modified by the CounterScene authors, 2026.

from setuptools import setup, find_packages

# read the contents of your README file
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    lines = f.readlines()

# remove images from README
lines = [x for x in lines if ".png" not in x]
long_description = "".join(lines)

setup(
    name="counterscene",
    packages=[package for package in find_packages() if package.startswith("ccdiff")],
    install_requires=[
        "torch==1.13.1",
        "torchvision==0.14.1",
    ],
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3.8",
    version="0.1.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
