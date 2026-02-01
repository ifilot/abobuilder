from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("abobuilder")
except PackageNotFoundError:
    __version__ = "0.0.0"  # optional fallback