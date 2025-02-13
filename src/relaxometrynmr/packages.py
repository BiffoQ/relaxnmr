# exports.py

import sys
import pkg_resources
import importlib

# Import all functions from core.py
from core import T1Functions

def get_package_version_info():
    """Get and print version information for all required packages."""
    packages = {
        'nmrglue': None,
        'numpy': None,
        'mrsimulator': None,
        'scipy': None,
        'matplotlib': None
    }
    
    print("Environment Information:")
    print("-" * 50)
    print(f"Python Version: {sys.version.split()[0]}")
    print("\nPackage Versions and Dependencies:")
    print("-" * 50)
    
    for package_name in packages:
        try:
            # Try to get package metadata
            package = pkg_resources.working_set.by_key[package_name]
            version = package.version
            deps = [str(r) for r in package.requires()]
            
            print(f"\n{package_name}:")
            print(f"Version: {version}")
            
            if deps:
                print("Dependencies:")
                for dep in deps:
                    print(f"  - {dep}")
            else:
                print("Dependencies: None found")
                
        except KeyError:
            try:
                # Alternative method using importlib
                module = importlib.import_module(package_name)
                version = getattr(module, '__version__', 'Version not found')
                print(f"\n{package_name}:")
                print(f"Version: {version}")
                print("Dependencies: Information not available")
            except ImportError:
                print(f"\n{package_name}:")
                print("Status: Package not found")

# Export the T1Functions class
__all__ = ['T1Functions']

# Print package information when the module is imported
print("\nNMR Processing Package Information")
print("=" * 50)
get_package_version_info()

# Example usage
if __name__ == "__main__":
    print("\nExample Usage:")
    print("-" * 50)
    print("from exports import T1Functions")
    print("t1 = T1Functions('path/to/data')")
    print("\nThis will give you access to all T1 processing functions while")
    print("maintaining a record of the package versions used in your analysis.")