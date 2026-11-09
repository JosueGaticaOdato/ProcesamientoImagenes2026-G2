# Guía de ejecución

Esta guía explica cómo ejecutar `main.py`, qué parámetros acepta y cómo combinar sus modos de seguimiento y apariencia.

## Antes de ejecutar

Abrir una terminal en la carpeta `TPFinal` y utilizar el entorno Python donde estén instaladas las dependencias del proyecto.

- El video de entrada configurado es `video_recortado.mp4`. Debe estar disponible en el directorio de ejecución.
- El detector configurado es YOLOv8s (`yolov8s.pt`).
- OSNet se utiliza automáticamente si existe `modelos/osnet_x025_msmt17.pth`.
- Para usar OCR se necesita EasyOCR en el mismo entorno:

```powershell
python -m pip install easyocr
```

EasyOCR puede descargar sus modelos al inicializarse por primera vez. Si no puede inicializarse por los errores contemplados en el código, el programa avisa y continúa sin lectura de dorsales.

Los ejemplos siguientes asumen la configuración actual: `UNIFORMES_AZUL_BLANCO = True`, calibrada para este partido.

## Ejecución por defecto

```powershell
python main.py
```

Procesa **12 segundos desde el segundo 23** del video original, hasta aproximadamente el segundo 35 o hasta el final del archivo si termina antes.

Utiliza YOLO y ByteTrack, clasificación de uniformes por color, seguimiento temporal y OSNet si están sus pesos locales. OCR y SigLIP quedan desactivados.

La salida se llama `diagnostico_tracking.mp4`.

## Parámetros disponibles

| Parámetro | Valor por defecto | Qué hace | Ejemplo |
|---|---|---|---|
| `--inicio NUMERO` | `23` | Segundo del video original desde el que comienza. Acepta decimales y debe ser mayor o igual a cero. | `--inicio 23.5` |
| `--segundos NUMERO` | `12` | Duración del segmento. Acepta decimales; `0` procesa hasta el final desde el inicio elegido. | `--segundos 60` |
| `--salida RUTA` | `diagnostico_tracking.mp4` | Nombre o ruta del MP4 generado. | `--salida resultado.mp4` |
| `--verificar-dorsales` | Desactivado | Activa OCR para intentar leer dorsales y confirmarlos mediante consenso temporal. | `--verificar-dorsales` |
| `--sin-ocr` | No indicado | Fuerza desactivar OCR. Con la configuración actual ya está desactivado por defecto. | `--sin-ocr` |
| `--sin-reid` | No indicado | Desactiva OSNet. Sin SigLIP, quedan los descriptores de color y textura junto con las otras pistas de seguimiento. | `--sin-reid` |
| `--siglip` | Desactivado | Activa la alternativa de apariencia profunda con SigLIP y desactiva OSNet en este modo de ejecución. | `--siglip` |
| `-h` o `--help` | — | Muestra la ayuda de los argumentos y termina. | `--help` |

Las opciones de activación se escriben solas: por ejemplo, `--verificar-dorsales`, sin agregar `True`. El orden de los parámetros no importa.

## Ejemplos de ejecución

### 1. Elegir inicio, duración y salida

```powershell
python main.py --inicio 23 --segundos 30 --salida resultado_30s.mp4
```

Procesa del segundo 23 al 53, sin superar el final del archivo. Mantiene los modos por defecto.

### 2. Ejecutar con OCR de dorsales

```powershell
python main.py --inicio 23 --segundos 30 --verificar-dorsales --salida resultado_con_ocr.mp4
```

Agrega lectura de números de camiseta. Intenta hasta dos recortes cada 30 frames y exige tres lecturas vigentes coincidentes antes de confirmar un dorsal. Puede aumentar considerablemente el tiempo de procesamiento en CPU.

Activar OCR no garantiza leer todos los números: deben verse con suficiente tamaño, nitidez y confianza. El seguimiento continúa aunque no se consiga una lectura. El ID interno mostrado sigue siendo distinto del dorsal.

### 3. Ejecutar explícitamente sin OCR

```powershell
python main.py --inicio 23 --segundos 30 --sin-ocr --salida resultado_sin_ocr.mp4
```

Desactiva la lectura de dorsales y conserva OSNet si están sus pesos. Actualmente equivale al modo por defecto respecto del OCR.

### 4. Ejecutar sin OSNet y sin OCR

```powershell
python main.py --inicio 23 --segundos 30 --sin-reid --sin-ocr --salida resultado_color_textura.mp4
```

Permite comparar el aporte de OSNet. Mantiene detección, ByteTrack, clasificación por color, descriptores de color/textura, compensación de cámara y asociación temporal.

### 5. Ejecutar con OCR, pero sin OSNet

```powershell
python main.py --inicio 23 --segundos 30 --sin-reid --verificar-dorsales --salida resultado_ocr_sin_reid.mp4
```

Combina color/textura con lectura de dorsales, sin extraer vectores OSNet.

### 6. Ejecutar con SigLIP

```powershell
python main.py --inicio 23 --segundos 30 --siglip --sin-ocr --salida resultado_siglip.mp4
```

Usa SigLIP como alternativa de apariencia profunda. Requiere sus dependencias y acceso a los pesos de `google/siglip-base-patch16-224`, descargándolos si no están disponibles en caché. Puede ser mucho más lento en CPU.

Con la configuración calibrada actual, el equipo sigue clasificándose por las reglas de color. Esta opción por sí sola no habilita el aprendizaje UMAP ni el agrupamiento automático.

### 7. Ejecutar con SigLIP y OCR

```powershell
python main.py --inicio 23 --segundos 30 --siglip --verificar-dorsales --salida resultado_siglip_ocr.mp4
```

Activa ambas etapas. OSNet queda desactivado y se agrega el costo de SigLIP y OCR.

### 8. Procesar hasta el final desde un instante

```powershell
python main.py --inicio 23 --segundos 0 --salida resultado_hasta_final.mp4
```

Procesa desde el segundo 23 hasta el final. `--segundos 0` no significa generar un video de cero segundos.

### 9. Procesar todo el video desde el comienzo

```powershell
python main.py --inicio 0 --segundos 0 --salida resultado_completo.mp4
```

Se pueden agregar `--verificar-dorsales`, `--sin-reid` o `--siglip` según el modo deseado. Las reglas de uniformes siguen siendo las configuradas para este partido.

### 10. Guardar en una carpeta específica

Crear primero la carpeta de destino. En PowerShell:

```powershell
New-Item -ItemType Directory -Force -Path "resultados"
python main.py --inicio 23 --segundos 30 --verificar-dorsales --salida "resultados/partido con OCR.mp4"
```

Si la ruta tiene espacios, se escribe entre comillas. El directorio padre debe existir antes de que el programa intente abrir el video de salida.

## Combinaciones y restricciones

| Combinación | Comportamiento |
|---|---|
| Sin opciones de apariencia | OSNet si existen sus pesos; OCR apagado. |
| `--verificar-dorsales` | Agrega OCR al modo visual elegido. |
| `--sin-reid` | Desactiva OSNet; no desactiva OCR ni SigLIP. |
| `--siglip` | Activa SigLIP y desactiva OSNet. |
| `--siglip --sin-reid` | Válida, pero `--sin-reid` es redundante porque SigLIP ya desactiva OSNet. |
| `--sin-ocr --verificar-dorsales` | Inválida: el programa informa que son incompatibles. |

Inicio y duración negativos también generan un error de argumentos. Un inicio fuera de la duración del video impide procesarlo.

## Archivos de salida

Para esta ejecución:

```powershell
python main.py --segundos 30 --salida resultado_con_ocr.mp4 --verificar-dorsales
```

Se generan:

```text
resultado_con_ocr.mp4
resultado_con_ocr/
    frame_XXXXXX.jpg
    ultimo_frame.jpg
    metricas.csv
    trayectorias.csv
```

- **MP4:** video con recuadros y etiquetas.
- **Capturas:** imágenes periódicas para inspección visual.
- **`metricas.csv`:** conteos y tiempos por etapa.
- **`trayectorias.csv`:** IDs, equipos y posiciones por frame.

Los CSV se guardan si hay registros para escribir. Usar un nombre de salida distinto para cada prueba: repetirlo puede sobrescribir resultados y reutiliza la misma carpeta de diagnóstico.

## Opciones que requieren editar el código

No existen argumentos de terminal para todos los ajustes. En `main.py` se pueden modificar:

| Configuración | Qué permite cambiar |
|---|---|
| `VIDEO` | Archivo de entrada; actualmente no existe `--video`. |
| `YOLO_IMGSZ` | Resolución de inferencia del detector. |
| `CONFIANZA_TRACKING` | Confianza mínima de detecciones pasadas al tracking. |
| `INTERVALO_OCR` y `MAX_OCR_POR_TURNO` | Frecuencia y presupuesto de OCR. |
| `INTERVALO_REID` y `INTERVALO_SIGLIP` | Frecuencia de extracción de apariencia profunda. |
| `FILTRAR_PISO_CANCHA` | Filtro de personas según el piso estimado. |
| `UNIFORMES_AZUL_BLANCO` | Reglas calibradas o modo de aprendizaje automático. |
| `MOSTRAR_VISTA_PREVIA` | Ventana de visualización, si OpenCV tiene soporte gráfico. |

Los umbrales de ByteTrack están en `bytetrack_jugadores.yaml`. Los votos de confirmación y las reglas de asociación están en `seguimiento.py`.

Cambiar `UNIFORMES_AZUL_BLANCO` a `False` habilita el modo genérico, que requiere SigLIP y aprendizaje de uniformes. Las descripciones de clasificación por color de los ejemplos anteriores corresponden al valor actual `True`.

Para una explicación de las técnicas, módulos y parámetros internos, consultar [EXPLICACION_PROYECTO.md](EXPLICACION_PROYECTO.md). Los resultados y límites de la validación se documentan en [VALIDACION_IDS.md](VALIDACION_IDS.md).

## Cómo comparar pruebas

Mantener el mismo inicio y duración, cambiar una opción por vez y guardar salidas con nombres distintos. Revisar los mismos jugadores y sus dorsales en cada resultado, además de los tiempos de procesamiento.

Si se evalúa la recuperación después de un corte de cámara, comenzar antes del corte para que el sistema acumule las identidades. Los conteos de IDs o recuperaciones no equivalen a exactitud de identidad.
