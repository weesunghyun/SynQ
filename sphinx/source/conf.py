import os
import sys

sys.path.insert(0, os.path.abspath('../../src'))
print(sys.path)

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SynQ'
copyright = '2025, Minjun Kim'
author = 'Minjun Kim'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # 반드시 포함되어야 함
    'sphinx.ext.napoleon'  # Google/Numpy 스타일의 docstring 지원
]

templates_path = ['_templates']
exclude_patterns = [
    '../../src/pytorchcv/**',
    '../../src/data_generate/pytorchcv/**'
]

autodoc_mock_imports = ['distill_data']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
import sphinx_rtd_theme

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ['_static']
