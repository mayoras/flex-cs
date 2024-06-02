"""
Copyright (C) 2024  Instituto Andaluz Interuniversitario en Ciencia de Datos e Inteligencia Computacional (DaSCI)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from setuptools import find_packages, setup

setup(
    name="flexCS",
    version="0.0.1",
    author="Cesar Mayora",
    author_email="cesarmay@protonmail.com",
    keywords="client-selection FL federated-learning flexible ML Machine-Learning AI",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mayoras/flex-cs",
    packages=find_packages(),
    install_requires=["flexible-fl", "matplotlib"],
    extras_require={
        "develop": ["pytest", "pytest-cov", "pytest-xdist", "coverage", "jinja2"]
    },
    python_requires=">=3.8.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Build Tools",
        # Pick your license as you wish
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate you support Python 3. These classifiers are *not*
        # checked by 'pip install'. See instead 'python_requires' below.
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3 :: Only",
    ],
)
