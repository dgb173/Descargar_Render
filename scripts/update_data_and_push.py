"""
Ejecuta el scraper principal para regenerar data.json y, si hay cambios,
los sube al repositorio git remoto.

Uso:
    python scripts/update_data_and_push.py

Requisitos:
    - Tener configurado git con las credenciales necesarias para realizar push.
    - Ser ejecutado desde un entorno donde `python run_scraper.py` funciona
      (por ejemplo, tu máquina local con Chrome headless instalado).
"""

from __future__ import annotations

import datetime
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON_BIN = sys.executable


def run_command(command: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    """Ejecuta un comando en la raíz del proyecto y devuelve el resultado."""
    print(f"→ Ejecutando: {' '.join(command)}")
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=False,
        text=True,
        check=check,
    )


def run_scraper():
    """Lanza el scraper asíncrono principal."""
    run_command([PYTHON_BIN, "run_scraper.py"])


def has_pending_changes() -> bool:
    """Devuelve True si hay cambios pendientes en data.json."""
    try:
        result = subprocess.run(
            ["git", "status", "--short", "data.json"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(result.stdout.strip())
    except subprocess.CalledProcessError as exc:
        print("No se pudo comprobar el estado de git:", exc)
        return False


def commit_and_push():
    """Añade data.json al staging, crea un commit y realiza push."""
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    commit_message = f"chore: actualizar data.json ({timestamp})"

    run_command(["git", "add", "data.json"])

    try:
        run_command(["git", "commit", "-m", commit_message])
    except subprocess.CalledProcessError as exc:
        print("No se pudo crear el commit (¿quizá no hay cambios?).")
        print(exc)
        return

    try:
        run_command(["git", "push"])
    except subprocess.CalledProcessError as exc:
        print("El push falló. Revisa tus credenciales de git o vuelve a intentar manualmente.")
        print(exc)


def main():
    run_scraper()

    if not has_pending_changes():
        print("No hay cambios en data.json después de ejecutar el scraper.")
        return

    commit_and_push()


if __name__ == "__main__":
    main()
