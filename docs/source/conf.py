# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ABO Builder'
copyright = '2026, Ivo Filot'
author = 'Ivo Filot'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_title = "ABO Builder User Manual"
html_logo = "_static/img/abobuilder-logo-128.png"
html_static_path = ['_static']

html_css_files = [
    'custom.css',
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css"
]

html_context = {
    "display_github": True,
    "github_user": "ifilot",
    "github_repo": "abobuilder",
    "github_version": "master",
    "conf_py_path": "/docs/source/",
}

html_theme_options = {
    "style_external_links": True,
}