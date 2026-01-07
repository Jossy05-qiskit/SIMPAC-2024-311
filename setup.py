from setuptools import setup, find_packages

setup(
    name='AI Driven Robotic Palletizing',
    version='0.1.0',
    author='TCHIBINDA MAKOMPA Romael Jossy',
    author_email='jossytchibinda@gmail.com',
    description='Reinforcement learning-based optimizer for 3D bin packing problem',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',

    packages=find_packages(),
    install_requires=[
        'numpy==1.26.4',
        'matplotlib==3.9.0',
        'tensorflow==2.10.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='==3.10',
)