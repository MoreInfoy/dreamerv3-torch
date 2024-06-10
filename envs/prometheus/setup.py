#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from setuptools import find_packages, setup

setup(
    name="envs.prometheus",
    version="0.0",
    packages=find_packages(),
    author="CLEAR Lab",
    maintainer="Shunpeng Yang",
    maintainer_email="syangcp@connect.ust.hk",
    license="MIT",
    description="Learn policy for the multi tasks of humanoid robot",
    python_requires=">=3.6",
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "isaacgym",
        "hydra-core",
        "torchvision",
        "numpy",
        "GitPython",
        "onnx",
        "mujoco"
    ],
)
