from setuptools import setup

__version__ = '0.3.0'

setup(
    name='rouge',
    version=__version__,
    description='An implementation of ROUGE family metrics for automatic summarization',
    url='https://github.com/neural-dialogue-metrics/rouge.git',
    author='cgsdfc',
    author_email='cgsdfc@126.com',
    keywords=[
        'NL', 'CL', 'MT',
        'natural language processing',
        'computational linguistics',
        'automatic summarization',
    ],
    packages=[
        'rouge',
        'rouge.tests',
    ],
    package_data={
        'rouge.tests': ['data/*'],
    },
    scripts=[
        'example.py',
        'rouge_score.py',
    ],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Topic :: Text Processing :: Linguistic',
        "Operating System :: OS Independent",
    ],
    license='LICENCE.txt',
    long_description=open('README.md').read(), install_requires=['numpy']
)
