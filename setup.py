import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="stats-toolkit-python",
    version="0.0.1",
    author="Nazly Sabbour",
    author_email="nazly.sabbour@gmail.com",
    description="code snippets for statistical tests in python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nazlysabbour1/stats-toolkit-python",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=["numpy==1.19.5",
                      "scipy==1.5.4", "pandas==1.1.5",
                      "statsmodels==0.12.2"]
)
