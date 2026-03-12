#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import shutil
from pathlib import Path
from datetime import datetime
from PIL import Image

from tracking_agujas import determineNeedle


BASE_DEBUG_DIR = Path("TesteoImagenes")
AGUJAS_TMP_DIR = Path("Agujas_resultados")


def unique_dir(base: Path) -> Path:
    """
    Dado un Path base (ej: TesteoImagenes/001_1),
    si ya existe, crea TesteoImagenes/001_1_1, 001_1_2, ...
    """
    if not base.exists():
        return base
    i = 1
    while True:
        candidate = base.parent / f"{base.name}_{i}"
        if not candidate.exists():
            return candidate
        i += 1


def main():
    cfg_path = "config.json"
    if len(sys.argv) > 1:
        img_paths = [Path(p) for p in sys.argv[1:]]
    else:

        img_paths = [Path(f"{i}.png") for i in range(1, 9)]


    img_paths = [p for p in img_paths if p.exists()]

    if not img_paths:
        print("No encontré imágenes. Pon 1.png, 2.png, ... en esta carpeta o pásalas por argumento.")
        return

    # Crear carpeta base de testeo
    BASE_DEBUG_DIR.mkdir(exist_ok=True)

    prev_value = 0.0  # valor inicial del dial

    print("==================================================")
    print("    TEST LOCAL DE AGUJAS (determineNeedle)        ")
    print("==================================================")
    print(f"Usando config: {cfg_path}")
    print(f"Imágenes: {[str(p) for p in img_paths]}")
    print("Debug por imagen en carpeta:", BASE_DEBUG_DIR)
    print("--------------------------------------------------\n")

    for idx, img_path in enumerate(img_paths, start=1):
        print("\n==================================================")
        print(f" IMAGEN {idx}: {img_path}")
        print("==================================================")

        # Cargar imagen
        img = Image.open(img_path).convert("RGB")

        if AGUJAS_TMP_DIR.exists():
            shutil.rmtree(AGUJAS_TMP_DIR)

        # Llamar a la función principal de agujas
        value = determineNeedle(img, cfg_path=cfg_path, prev_value=prev_value)

        try:
            value_f = float(value)
        except Exception:
            value_f = value

        print(f"\n[RESULTADO] prev={prev_value:.3f}  ->  valor_actual={value_f:.3f}")


        run_base = BASE_DEBUG_DIR / f"{idx:03d}_{img_path.stem}"
        run_dir = unique_dir(run_base)

        if AGUJAS_TMP_DIR.exists():

            shutil.move(str(AGUJAS_TMP_DIR), str(run_dir))
        else:

            run_dir.mkdir(parents=True, exist_ok=True)

        resumen_path = run_dir / "resultado.txt"
        with open(resumen_path, "w", encoding="utf-8") as f:
            f.write(f"Imagen: {img_path}\n")
            f.write(f"Fecha/hora test: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"prev_value (entrada): {prev_value:.6f}\n")
            f.write(f"valor_actual (salida determineNeedle): {value_f:.6f}\n")

        print(f"[DEBUG] Carpeta de esta imagen: {run_dir}")


        try:
            prev_value = float(value_f)
        except Exception:

            print("[WARN] No pude convertir valor_actual a float, se mantiene prev_value.")
            pass


if __name__ == "__main__":
    main()
