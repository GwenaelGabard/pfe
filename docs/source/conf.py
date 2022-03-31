import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


project = 'pfe'
copyright = '2022, Gwénaël Gabard'
author = 'Gwénaël Gabard'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.mathjax', 'sphinxcontrib.bibtex']
bibtex_bibfiles = ['references.bib']

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'
