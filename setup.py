from setuptools import find_packages, setup
from typing import List


HYPHEN_E_DOT = '-e .'

def get_requirements(file_path) -> List[str]:
    """
    This function returns a list of requirements from the given file path.
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name='Generic-ML-project',
    version='0.0.1',
    author='Ilvan',
    author_email='ilyasrachid00@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    description='A generic ML project template',
)