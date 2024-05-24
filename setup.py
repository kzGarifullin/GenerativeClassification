from setuptools import setup, find_packages

with open('requirements.txt', encoding='utf-8') as file:
    requirements = file.read().splitlines()

with open('README.md', encoding='utf-8') as file:
    long_description = file.read()

setup(
    name='GenerativeClassification',
    version='0.0.0',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/kzGarifullin/GenerativeClassification',
    packages=find_packages(include=('GenerativeClassification',)),
    python_requires='>=3.10',
    install_requires=requirements,
)
