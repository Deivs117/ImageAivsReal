"""Configuration module for pytest.

Adds the parent of the 'inference' package to sys.path so that tests
can import inference modules without requiring an installed package.
"""
import sys
import os


def add_inference_parent_to_sys_path():
    """Add the parent of the 'inference' package to sys.path.

    1) Checks the common path <repo_root>/service/inference.
    2) If not found, searches recursively for 'inference' folder
       and adds its parent.
    3) As a last resort, adds the repo root.

    This allows ``import inference...`` to work during pytest
    collection.
    """
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )

    # Preferred path: <repo_root>/service/inference
    service_parent = os.path.join(repo_root, "service")
    if os.path.isdir(os.path.join(service_parent, "inference")):
        if service_parent not in sys.path:
            sys.path.insert(0, service_parent)
        return

    # Recursive search: if any 'inference' folder exists, add its parent
    for dirpath, dirnames, _ in os.walk(repo_root):
        if "inference" in dirnames:
            # dirpath is the parent that contains 'inference'
            parent = dirpath
            if parent not in sys.path:
                sys.path.insert(0, parent)
            return

    # Last resort: add the repo root
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


# Run when conftest.py is imported (pytest loads this before collection)
add_inference_parent_to_sys_path()
