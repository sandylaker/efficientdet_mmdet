import setuptools

setuptools.setup(
    name="effdet_mmdet",
    version="0.0.1",
    author="Yawei Li",
    author_email="yawei@tum.de",
    description="EfficientDet based on mmdetection",
    packages=setuptools.find_packages(exclude=['tests', 'tools', 'configs']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)