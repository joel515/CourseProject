from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='cplsa',
    version='0.0.1',
    author='Joel Kopp',
    author_email='joelk2@illinois.edu',
    url='https://github.com/joel515/CourseProject',
    description='Context PLSA Topic Mining',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)