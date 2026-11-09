# TPFinal — Versión 2

Versión del seguimiento de jugadores con YOLOv8s, ByteTrack, clasificación HSV, asociación temporal, compensación de cámara y apariencia OSNet. Incluye OCR opcional de dorsales y la alternativa SigLIP.

## Preparación

Desde esta carpeta, en Windows con Python instalado:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Para habilitar OCR, SigLIP y el aprendizaje automático de uniformes, instalar también las dependencias opcionales:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-opcionales.txt
```

Las dependencias indican los paquetes necesarios; no constituyen un entorno bloqueado a versiones exactas. La instalación de PyTorch debe ser compatible con el equipo si se desea usar GPU.

## Video de entrada

Colocar en esta carpeta el archivo **`video_recortado.mp4`** utilizado por el proyecto. Los videos no se incluyen en esta copia ni se versionan. Para usar otro archivo, modificar `VIDEO` en `main.py`; las reglas de color y cancha actuales están calibradas para el partido original, con equipo A azul y B blanco.

No reemplazarlo por otro recorte suponiendo que conserva los mismos tiempos: el segundo inicial se refiere al archivo de entrada concreto.

## Ejecutar

Prueba por defecto: 12 segundos desde el segundo 23, sin OCR y con OSNet.

```powershell
.\.venv\Scripts\python.exe main.py
```

Con OCR:

```powershell
.\.venv\Scripts\python.exe main.py --inicio 23 --segundos 30 --verificar-dorsales --salida resultado_con_ocr.mp4
```

Hasta el final, desde el segundo 23:

```powershell
.\.venv\Scripts\python.exe main.py --inicio 23 --segundos 0 --salida resultado_completo.mp4
```

Se incluyen `yolov8s.pt` y `modelos/osnet_x025_msmt17.pth` para conservar los modelos locales de esta versión. La arquitectura OSNet, su licencia y la procedencia de sus pesos están en `vendor/`. EasyOCR y SigLIP pueden descargar sus propios modelos al utilizarse por primera vez.

## Documentación

- [Guía de ejecución y parámetros](GUIA_EJECUCION.md).
- [Explicación del código y técnicas aplicadas](EXPLICACION_PROYECTO.md).
- [Notas técnicas del seguimiento](TRACKING.md).
- [Resultados históricos y límites de validación](VALIDACION_IDS.md).

La auditoría describe ejecuciones anteriores; sus videos, recortes y tablas de evidencia no forman parte de este paquete. `comparar_referencias.py` se incluye como herramienta, pero requiere las referencias manuales externas descritas en esa auditoría.

## Pruebas

```powershell
.\.venv\Scripts\python.exe -m unittest test_seguimiento test_apariencia test_dorsales test_identidad_estable test_reid_ligero
```

## Archivos versionados

Se incluyen el programa principal, sus módulos, la configuración ByteTrack, modelos locales, pruebas y documentación. Se excluyen entornos virtuales, cachés, videos, capturas, CSV y resultados generados mediante `.gitignore`.

Los IDs y las recuperaciones son estimaciones del sistema: las validaciones históricas todavía muestran intercambios de identidad. Esta copia conserva el comportamiento actual del código.
