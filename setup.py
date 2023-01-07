# MIT License
#
# Copyright (c) 2019 Cong Feng
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from setuptools import setup

__version__ = "0.3.0"

setup(
    name="rouge",
    version=__version__,
    description="An implementation of ROUGE family metrics for automatic summarization",
    url="https://github.com/neural-dialogue-metrics/rouge.git",
    author="Cong Feng",
    author_email="cgsdfc@126.com",
    keywords=[
        "NL",
        "CL",
        "MT",
        "natural language processing",
        "computational linguistics",
        "automatic summarization",
    ],
    packages=[
        "rouge",
        "rouge.tests",
    ],
    package_data={
        "rouge.tests": ["data/*"],
    },
    scripts=[
        "scripts/example.py",
        "scripts/rouge_score.py",
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Text Processing :: Linguistic",
        "Operating System :: OS Independent",
    ],
    license="LICENCE.txt",
    long_description=open("README.md").read(),
    install_requires=["numpy"],
)
