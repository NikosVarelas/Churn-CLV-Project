import os
from setuptools import setup, find_packages, Command


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import subprocess
        subprocess.Popen('rm -rf ./build ./*.pyc ./*.egg-info', shell=True).wait()


requirements = ''
with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'requirements.txt')) as f:
    requirements = f.readlines()

setup(
    name='churnclv',
    description="Bundle of models and preprocessing techniques for Churn and CLV predictions.",
    author='Nikos Varelas',
    author_email='nikos.vare@gmail.com',
    version='0.1.0',
    url='https://github.com/NikosVarelas/Churn-CLV-Project',
    install_requires=requirements,
    platforms='any',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    cmdclass={
        'clean': CleanCommand,
    },
    packages=find_packages(include=['churnclv*']),
)