from setuptools import setup, find_packages
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent
root_package = "images_pipeline/"
root_package_dir = "src/."
setup(
    name="images_pipeline",
    version="0.0.1",
    description="A package to process images",
    long_description="",
    long_description_content_type="text/markdown",
    url="https://github.com/JonathanHidalgoN",
    author="Jonathan Hidalgo",
    author_email="ja.hidalgonunez@ugto.mx",
    license="MIT",
    classifiers=["Programming Language :: Python :: 3"],
    python_requires=">=3.6",
    packages=["src"],
    package_dir={"": root_package_dir}
)
