from setuptools import setup, find_packages
with open("README.md") as fh:
    long_description = fh.read()
setup(
    name='OCBO',
    version='0.1.2',
    packages=find_packages(include=['OCBO', 'OCBO.*']),
    url = 'https://github.com/zhaofeng-shu33/OCBO',
    author = 'zhaofeng-shu33',
    author_email = '616545598@qq.com',
    long_description = long_description,
    long_description_content_type="text/markdown",
    install_requires=['dragonfly-opt', 'scikit-learn', 'matplotlib'],
    license = 'MIT License',
    description = 'Offline Contextual Bayesian Optimization'    
)
