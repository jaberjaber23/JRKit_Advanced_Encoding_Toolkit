from setuptools import setup, find_packages

setup(
    name='JrKit',
    version='1.0.0',
    author='Jaber Jaber',
    author_email='Jaberib647@gmail.com',
    description='JR\'s Advanced Encoding Toolkit: Innovative encoding methods for data representation and feature extraction',
    url='https://github.com/jaberjaber23/JR_Advanced_Encoding_Toolkit',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='encoding machine-learning data-science toolkit',
    install_requires=[
        'numpy',
        'pandas',
    ],
)
