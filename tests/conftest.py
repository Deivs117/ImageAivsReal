import sys
import os

def add_inference_parent_to_sys_path():
    """
    Añade a sys.path la carpeta que contiene el paquete 'inference'.
    1) Comprueba la ruta común <repo_root>/service/inference (la que tú indicastes).
    2) Si no existe, busca recursivamente una carpeta llamada 'inference' y añade su padre.
    3) Como último recurso añade la raíz del repo.
    Esto permite que `import inference...` funcione durante la recolección de pytest.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Ruta preferida: <repo_root>/service/inference
    service_parent = os.path.join(repo_root, "service")
    if os.path.isdir(os.path.join(service_parent, "inference")):
        if service_parent not in sys.path:
            sys.path.insert(0, service_parent)
        return

    # Búsqueda recursiva: si hay cualquier carpeta 'inference', añade su padre
    for dirpath, dirnames, _ in os.walk(repo_root):
        if "inference" in dirnames:
            parent = dirpath  # dirpath contiene la carpeta 'inference'
            if parent not in sys.path:
                sys.path.insert(0, parent)
            return

    # Último recurso: añadir la raíz del repo
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

# Ejecutar al importar conftest.py (pytest lo carga antes de la recolección)
add_inference_parent_to_sys_path()