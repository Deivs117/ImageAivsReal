import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "proto", "generated", "service"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
    
def add_package_parent_to_sys_path(package_name="inference", max_up_levels=10):
    """
    Busca hacia arriba a partir de tests/ hasta encontrar un directorio llamado `package_name`.
    Si lo encuentra, añade su padre a sys.path (para que `import inference` funcione).
    Si no lo encuentra, intenta la ruta fallback proto/generated/service.
    Finalmente, añade la raíz del repo (tests/..) como último recurso.
    """
    cur = os.path.abspath(os.path.dirname(__file__))

    # Subir hasta max_up_levels carpetas buscando "<cur>/.../<package_name>"
    for _ in range(max_up_levels):
        candidate = os.path.join(cur, package_name)
        if os.path.isdir(candidate):
            parent = cur
            if parent not in sys.path:
                sys.path.insert(0, parent)
            return
        parent_dir = os.path.dirname(cur)
        if parent_dir == cur:
            break
        cur = parent_dir

    # Fallback específico si tu package está en proto/generated/service/inference
    fallback = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "proto", "generated", "service"))
    if os.path.isdir(os.path.join(fallback, package_name)):
        if fallback not in sys.path:
            sys.path.insert(0, fallback)
        return

    # Último recurso: añadir la raíz del repo (tests/..)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

# Ejecuta la función al importar conftest.py (pytest lo carga antes de la recolección)
add_package_parent_to_sys_path("inference")