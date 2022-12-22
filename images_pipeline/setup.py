from setuptools import setup, find_packages
import pathlib

# The directory containing this file
here = pathlib.Path(__file__).parent.resolve()
root_package = "images_pipeline"
root_package_dir = "src/"
# The text of the README file
setup(
    name="images_pipeline",
    version="0.1.0",
    description="A small package",
    url="https://github.com/JonathanHidalgoN",
    author="Jonathan Hidalgo",
    author_email="ja.hidalgonunez@ugto.mx",
    classifiers=["Programming Language :: Python :: 3"],
    packages=[root_package]
    + [
        f"{root_package}.{item}"
        for item in find_packages(where=root_package_dir + root_package)
    ],
    package_dir={"": "src"},
    python_requires=">=3.6, <4",
)