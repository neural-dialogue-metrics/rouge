from setuptools import setup

setup(
    name='rouge',
    version='0.1',
    description='implementation of ROUGE family metrics for automatic summarization',
    url='https://github.com/neural-dialogue-metrics/rouge.git',
    author='cgsdfc',
    author_email='cgsdfc@126.com',
    keywords=[
        'NL', 'CL', 'MT',
        'natural language processing',
        'computational linguistics',
        'automatic summarization',
    ],
    package='rouge',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache-v2',
        'Programming Language :: Python :: 3',
        'Topic :: Text Processing :: Linguistic'
    ],
    license='LICENCE.txt',
    long_description=open('README.md').read(),
)
