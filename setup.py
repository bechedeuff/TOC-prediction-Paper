from setuptools import find_packages, setup
from typing import List

"""
Use pip install -r requirements.txt to install the requirements.
OBS: you need to activate the env created with python 3.11.4 (lasted cheked version).
and be in the folder where the requirements.txt is located in your cmd.
"""

var = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """
    this function will return the list of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if var in requirements:
            requirements.remove(var)

    return requirements


setup(
    name="TOC_Prediction",
    version="1.0",
    author="Bernardo Chede",
    author_email="bechede@id.uff.br",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
