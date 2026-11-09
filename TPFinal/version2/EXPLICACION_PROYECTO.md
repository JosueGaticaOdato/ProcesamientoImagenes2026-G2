# Seguimiento de jugadores: implementación y parámetros

Documento para presentar el estado actual del trabajo práctico. Describe el código existente, las decisiones aplicadas y los parámetros que se pueden modificar. La implementación principal está en `main.py`.

## 1. Objetivo del sistema

Procesar un video de básquet para detectar personas, distinguir equipos y mantener una identidad temporal para cada jugador. El resultado es un video con recuadros y etiquetas, acompañado por archivos CSV para analizar las trayectorias y el procesamiento.

Hay tres problemas diferentes:

- **Detección:** localizar una persona en un frame.
- **Clasificación de equipo:** decidir a qué equipo corresponde su uniforme.
- **Identidad:** intentar reconocer a la misma persona a lo largo del video, incluso después de una oclusión o pérdida de detección.

Detectar correctamente o acertar el equipo no garantiza conservar la identidad correcta.

## 2. Flujo general aplicado

```text
Video y selección del segmento
    → YOLOv8s: detección de personas + ByteTrack: seguimiento inicial
    → Detección de primer plano y compensación de cámara
    → Eliminación de cajas duplicadas y filtrado por cancha
    → Recortes de camiseta, cuerpo, cabeza, shorts y dorsal
    → Equipo por HSV + descriptores de apariencia
    → OSNet opcional / SigLIP alternativo / OCR opcional
    → Asociación temporal, confirmación y recuperación de identidades
    → Recuadros, video de salida, capturas y CSV
```

La configuración actual está adaptada al partido de `video_recortado.mp4`: equipo A azul, equipo B blanco y posibles árbitros de gris. Son reglas calibradas para este video.

## 3. Qué hace cada archivo importante

| Archivo | Responsabilidad |
|---|---|
| `main.py` | Configuración, argumentos de terminal, carga de modelos y coordinación de todo el procesamiento. También dibuja etiquetas y guarda resultados. |
| `apariencia.py` | Estimación del piso, reglas HSV de uniformes, recortes por región, descriptores de color/textura y lectura OCR con EasyOCR. |
| `seguimiento.py` | Asociación entre frames, confirmación de equipo e ID, memoria de apariencias, recuperación y predicción breve cuando falta una detección. |
| `movimiento_camara.py` | Estima el movimiento del fondo con ORB y RANSAC para compensar paneos y zoom. |
| `reid_ligero.py` | Carga OSNet x0.25 y obtiene vectores de apariencia del cuerpo completo. |
| `dorsales.py` | Reúne lecturas OCR de distintos frames y exige consenso antes de aceptar un dorsal. |
| `bytetrack_jugadores.yaml` | Umbrales y memoria del seguimiento inicial de ByteTrack. |
| `vendor/osnet.py` | Arquitectura OSNet utilizada por el extractor de apariencia. |
| `comparar_referencias.py` | Compara trayectorias generadas con muestras manuales de referencia. |
| `VALIDACION_IDS.md` | Registra la auditoría realizada, sus resultados y limitaciones. |



## 4. Técnicas y bloques principales

### 4.1 Detección con YOLOv8s y seguimiento con ByteTrack

En `procesar_video()`, `yolo.track(...)` procesa cada frame con `classes=[0]`: se busca la clase persona. `persist=True` conserva el estado del tracker entre frames.

YOLO entrega recuadros y confianza de detección. ByteTrack asocia detecciones y les asigna un ID inicial, llamado `raw_id` en el código. Se conservan detecciones de confianza baja para permitir la segunda asociación de ByteTrack, útil ante oclusiones.

El `raw_id` de ByteTrack y el ID final del proyecto son distintos. El módulo propio intenta conservar el ID final aunque cambie el de ByteTrack.

### 4.2 Filtrado de cancha y de personas

`posible_jugador()` descarta candidatos demasiado pequeños o fuera de la zona esperada. `obtener_contorno_cancha()` busca el color del parquet en HSV, limpia la máscara con una apertura morfológica y obtiene una envolvente convexa del contorno mayor.

Se comprueba la posición de los pies dentro de esa región y un parche debajo de ellos para distinguir parquet o pintura roja de publicidad y bandas laterales. Un posible árbitro gris fiable tiene una excepción al segundo filtro. Si no se detecta una región de piso suficientemente grande, el filtro por contorno se omite.

Las personas ya conocidas con continuidad geométrica tienen un tratamiento más tolerante para evitar cortar trayectorias por fallos breves del filtro.

### 4.3 Equipo mediante HSV

`recortar_camiseta()` extrae aproximadamente el torso: entre el 20 % y el 80 % del ancho de la caja y entre el 18 % y el 55 % de su altura. Se usa el frame original para conservar detalle.

`clasificar_uniforme_azul_blanco()` analiza tono, saturación y brillo. Evalúa proporciones de azul, blanco y gris; también revisa los shorts cuando están disponibles. Una combinación incompatible puede dejar el equipo sin confirmar.

HSV permite separar color y brillo mejor que comparar directamente los canales RGB, aunque sigue dependiendo de la iluminación. La categoría de árbitro es una heurística de vestimenta, no un detector entrenado de árbitros.

### 4.4 Apariencia por regiones

`recortes_persona()` obtiene cuerpo, cabeza, shorts y dorsal usando posiciones relativas dentro de la caja. Descarta regiones demasiado pequeñas y, salvo shorts, regiones con poca nitidez según la varianza del Laplaciano.

`descriptor_rapido()` divide cada recorte en tres franjas y combina histogramas HSV, histogramas de brillo y textura mediante el Laplaciano. Los vectores se normalizan para compararlos.

La cabeza aporta apariencia general, como pelo y zonas visibles del rostro. No se aplica reconocimiento facial especializado.

### 4.5 Reidentificación con OSNet

`Reidentificador.extraer()` redimensiona los cuerpos a 128 × 256 píxeles, normaliza los canales y extrae vectores de 512 componentes mediante OSNet x0.25 preentrenado en MSMT17.

Se activa automáticamente si existe `modelos/osnet_x025_msmt17.pth`. Se actualiza por ID cada diez frames, en lotes de hasta ocho recortes. Se evitan nuevas muestras con oclusión por solapamiento para reducir la contaminación de las referencias.

La similitud entre vectores ayuda a comparar personas después de perderlas. Los pesos son preentrenados: este código no entrena OSNet con los jugadores del partido.

### 4.6 Asociación y confirmación temporal

`SeguimientoJugadores.actualizar()` combina posición, movimiento, continuidad de ByteTrack, equipo, apariencia y dorsal cuando existe. Utiliza `linear_sum_assignment` para resolver la asignación entre trayectorias y detecciones a partir de una matriz de costos.

Una identidad nueva requiere una ventana de diez frames y al menos ocho votos para el equipo. Los cupos son cinco jugadores por equipo y dos posibles árbitros:

| Categoría | IDs internos reservados |
|---|---|
| Equipo A | 1 a 5 |
| Equipo B | 6 a 10 |
| Posibles árbitros | 11 a 12 |

Los cupos suponen los mismos participantes durante el segmento; no modelan sustituciones. Un ID interno no es el número de camiseta.

La galería conserva referencias para recuperar trayectorias perdidas. La recuperación exige compatibilidad y diferencia suficiente frente a otros candidatos; una coincidencia ambigua puede quedar sin ID. Las referencias iniciales OSNet se conservan como anclas para evitar que asociaciones dudosas las sustituyan gradualmente.

Se suprimen cajas muy solapadas del mismo equipo y se controla que un ID de ByteTrack no tenga varios propietarios confirmados.

### 4.7 Cámara, primeros planos y predicción

`MovimientoCamara.actualizar()` busca hasta 400 puntos ORB en el fondo de una imagen reducida a 480 píxeles de ancho. Compara descriptores y estima una transformación afín parcial con RANSAC. Se rechazan transformaciones sin suficientes correspondencias fiables o con cambios excesivos.

`compensar_camara()` transforma las cajas históricas antes de asociar posiciones: así el movimiento de la cámara no se interpreta completamente como movimiento de los jugadores.

La presencia de una persona con caja mayor al 18 % del área de imagen y altura mayor al 65 % activa la heurística de primer plano. Al entrar o salir de ese estado se invalidan posiciones anteriores y se conservan identidades para intentar recuperarlas. No es un detector general de todos los cortes de edición.

Si falta una detección, `predecir()` puede extrapolar su posición hasta cinco frames a partir de las últimas observaciones. La etiqueta indica `Predicho`; esa caja no constituye una detección nueva.

### 4.8 OCR de dorsales

OCR significa reconocimiento óptico de caracteres. `LectorDorsal` usa EasyOCR sobre el recorte del dorsal ampliado tres veces y permite únicamente dígitos.

Para aceptar una lectura exige un único candidato de uno o dos dígitos, confianza mínima de 0.85 y altura del texto de al menos el 20 % del recorte ampliado. El texto se conserva como cadena para distinguir `0` de `00`.

`ControlDorsales` exige que las últimas tres lecturas vigentes coincidan y provengan de frames separados. La ventana temporal es aproximadamente de cinco segundos. Una lectura aislada no confirma el dorsal.

Por defecto está desactivado. Al activarlo, se intentan hasta dos recortes cada 30 frames; no se lee a todas las personas en todos los frames. Si no hay consenso, continúa el seguimiento visual. Si EasyOCR no está disponible o falla su inicialización con los errores contemplados, el código avisa y continúa sin OCR.

### 4.9 Alternativa SigLIP, UMAP y agrupamiento

El código conserva una alternativa con SigLIP para obtener características profundas de las camisetas. `--siglip` la habilita y desactiva OSNet en la ejecución habitual por terminal.

Con `UNIFORMES_AZUL_BLANCO = True`, activar SigLIP **no activa el aprendizaje UMAP**: el equipo sigue clasificándose con las reglas calibradas.

Con `UNIFORMES_AZUL_BLANCO = False`, se habilita el aprendizaje automático de uniformes con SigLIP, reducción UMAP y agrupamiento. Se guardan resultados en `uniformes_cache.npz` para reutilizarlos si coinciden el video y la configuración relevante. La clasificación rápida posterior compara vectores y referencias de color; UMAP se utiliza en el aprendizaje inicial. El tercer grupo automático no garantiza representar árbitros.

## 5. Parámetros que se cambian desde la terminal

Se escriben después de `python main.py`; no requieren editar el código.

| Opción | Valor por defecto | Qué modifica |
|---|---|---|
| `--inicio` | `23` | Segundo del video original desde el que se comienza. |
| `--segundos` | `12` | Duración que se procesa; `0` continúa hasta el final del archivo desde el inicio elegido. |
| `--salida` | `diagnostico_tracking.mp4` | Ruta o nombre del video generado. |
| `--verificar-dorsales` | Desactivado | Activa los intentos OCR y el consenso de dorsales. |
| `--sin-ocr` | No indicado | Fuerza desactivar OCR; hoy resulta redundante con la configuración por defecto. |
| `--sin-reid` | No indicado | Desactiva OSNet para comparar usando las otras pistas visuales. |
| `--siglip` | Desactivado | Activa SigLIP como alternativa de apariencia; puede ser costoso en CPU. |

`--sin-ocr` y `--verificar-dorsales` son incompatibles. Inicio y duración no admiten valores negativos. En Python, los guiones de las opciones se convierten en guiones bajos: `--sin-ocr` se consulta como `opciones.sin_ocr`.

Ejemplo con OCR, desde el segundo 23 y durante 30 segundos:

```powershell
python main.py --inicio 23 --segundos 30 --verificar-dorsales --salida resultado_con_ocr.mp4
```

Ejemplo comparable sin OCR:

```powershell
python main.py --inicio 23 --segundos 30 --sin-ocr --salida resultado_sin_ocr.mp4
```

Ejemplo para medir el aporte de OSNet:

```powershell
python main.py --inicio 23 --segundos 30 --sin-reid --sin-ocr --salida resultado_sin_reid.mp4
```

La salida es opcional. Aunque arriba de `main.py` aparece `SALIDA = "resultado_equipos_mejorado.mp4"`, al ejecutar por terminal el parser reemplaza ese valor por `diagnostico_tracking.mp4` o por el argumento recibido.

## 6. Parámetros internos principales de `main.py`

Estos se modifican en el archivo; no todos tienen una opción de terminal.

| Parámetro | Valor actual | Efecto y criterio de ajuste |
|---|---|---|
| `VIDEO` | `video_recortado.mp4` | Archivo de entrada. No existe actualmente un argumento `--video`. |
| `YOLO_IMGSZ` | `640` | Tamaño de inferencia de YOLO. Subirlo puede ayudar con personas pequeñas y aumentar el costo. |
| `MODELO_YOLO` | `yolov8s.pt` | Pesos del detector utilizado. Cambiar modelo altera costo y calidad de detección. |
| `CONFIANZA_TRACKING` | `0.10` | Confianza mínima pasada a `yolo.track`. Subirla elimina detecciones débiles que ByteTrack podría aprovechar. |
| `CONFIANZA` | `0.25` | Umbral del detector en el aprendizaje automático de uniformes; no es el umbral principal del bucle de tracking. |
| `FILTRAR_PISO_CANCHA` | `True` | Habilita el filtro calibrado para el piso de este partido. Revisarlo al cambiar de cancha. |
| `UNIFORMES_AZUL_BLANCO` | `True` | Usa reglas calibradas; en `False` habilita el aprendizaje genérico y requiere SigLIP. |
| `USAR_OCR_DORSALES` | `False` | Estado inicial del OCR; las opciones de terminal pueden activarlo o desactivarlo. |
| `INTERVALO_OCR` | `30` frames | Menor intervalo: más intentos y costo; mayor intervalo: menos oportunidades de consenso. |
| `MAX_OCR_POR_TURNO` | `2` | Máximo de recortes OCR por turno; subirlo aumenta el trabajo de esa etapa. |
| `USAR_REID` | Existencia de los pesos | Habilita OSNet si está el archivo local, salvo desactivación por opciones. |
| `INTERVALO_REID` | `10` frames | Frecuencia de actualización de OSNet por ID. Menor valor aumenta el costo. |
| `INTERVALO_SIGLIP` | `15` frames | Frecuencia de actualización de SigLIP por ID, cuando está habilitado. |
| `HSV_BINS` | `(18, 8)` | Resolución del histograma de tono y saturación de camiseta. |
| `MOSTRAR_VISTA_PREVIA` | `False` | Activa la ventana de visualización si OpenCV tiene soporte gráfico. |
| `DEVICE` | CUDA si está disponible; si no, CPU | Selección automática para los modelos que usan esa variable. |

Los intervalos se expresan en frames: 30 frames equivalen a un segundo solamente en un video de 30 FPS. El intervalo regula cuándo se intenta el cálculo, no garantiza una lectura válida.

### Parámetros del modo de aprendizaje automático

| Parámetro | Valor | Uso |
|---|---|---|
| `SEGUNDOS_APRENDIZAJE` | `20` | Tiempo de video para recolectar muestras de uniformes. |
| `MAX_MUESTRAS` | `80` | Límite de muestras para aprender. |
| `UMAP_COMPONENTES` | `3` | Dimensión reducida de las características. |
| `UMAP_VECINOS` | `15` | Vecindario considerado por UMAP, limitado según las muestras disponibles. |
| `UMAP_DISTANCIA` | `0.0` | Parámetro `min_dist` de UMAP. |
| `SIMILITUD_MINIMA` | `0.50` | Similitud mínima para aceptar clasificación con referencias SigLIP. |
| `MARGEN_SIGLIP` | `0.01` | Diferencia mínima entre los dos equipos en esa clasificación. |

Estos parámetros de aprendizaje no gobiernan las reglas HSV del modo calibrado actual. También quedan constantes y funciones del clasificador anterior, como `FACTOR_DISTANCIA` y `clasificar_camiseta()`; no deben presentarse como controles activos del flujo rápido actual.

## 7. Parámetros de ByteTrack

Se encuentran en `bytetrack_jugadores.yaml`.

| Parámetro | Valor | Significado |
|---|---|---|
| `track_high_thresh` | `0.25` | Umbral para detecciones de alta confianza usadas en la primera asociación. |
| `track_low_thresh` | `0.10` | Umbral inferior de detecciones débiles para la segunda asociación. |
| `new_track_thresh` | `0.30` | Confianza requerida para iniciar una trayectoria de ByteTrack. |
| `track_buffer` | `90` | Memoria nominal de trayectorias perdidas; aproximadamente tres segundos a 30 FPS. |
| `match_thresh` | `0.80` | Umbral de asociación de ByteTrack; no es un porcentaje de exactitud. |
| `fuse_score` | `true` | Incorpora la confianza de detección al costo de asociación. |

La memoria de ByteTrack es distinta de la memoria del módulo propio de seguimiento.

## 8. Umbrales importantes dentro de los módulos

Son valores incrustados en funciones; cambiarlos exige editar esas funciones.

| Lugar | Valor actual | Qué controla |
|---|---|---|
| `main.py`, creación de `SeguimientoJugadores` | Ventana `10`; memoria `max(10, int(fps*3))` | Confirmación temporal y memoria espacial cercana a tres segundos. |
| `seguimiento.py`, confirmación nueva | `8` votos en ventana de `10` | Estabilidad necesaria antes de asignar una identidad nueva. |
| `seguimiento.py`, `predecir` | Hasta `5` frames | Duración máxima de extrapolación sin detección. |
| `seguimiento.py`, `indices_sin_duplicados` | IoU mayor a `0.65` | Supresión de cajas duplicadas con equipo compatible. |
| `main.py`, solapamiento | IoU mayor a `0.20` | Marca oclusión para evitar muestras de apariencia contaminadas. |
| `seguimiento.py`, `memoria_confiable` | Similitud OSNet `0.75`; margen `0.04` | Protege referencias ante candidatos del mismo equipo, con excepciones según las pistas disponibles. |
| `seguimiento.py`, `recuperar` | Costo menor a `0.32` con ReID o `0.15` sin él; margen mínimo `0.025` | Aceptación de la recuperación por galería frente a otros candidatos. Menor costo requerido es más estricto. |
| `apariencia.py`, filtro OCR | Confianza `0.85`; altura relativa `0.20` | Calidad mínima de una lectura individual. |
| `main.py`, creación de `ControlDorsales` | Edad `max(30, int(fps*5))`; separación `max(1, int(fps/10))` | Vigencia y separación temporal mínima de las lecturas OCR. |
| `dorsales.py`, `confirmado` | Últimas `3` lecturas vigentes iguales | Consenso previo a utilizar un dorsal. |
| `movimiento_camara.py` | `12` correspondencias; `10` inliers; proporción `0.40` | Evidencia mínima para aceptar la transformación estimada. |

IoU es el área de intersección de dos cajas dividida por su unión. Un valor alto indica gran solapamiento. Los umbrales de similitud y de costo tampoco son porcentajes de identidad correcta.

Para adaptar colores, se edita `clasificar_uniforme_azul_blanco()`: por ejemplo, azul usa tono entre 90 y 140, saturación mayor a 60 y brillo mayor a 30 en la escala HSV de OpenCV. No existe un único umbral universal para todos los partidos.

## 9. Qué se guarda y cómo se interpreta

Además del MP4 se crea una carpeta con el mismo nombre sin extensión. Allí se guardan capturas periódicas, `ultimo_frame.jpg`, `metricas.csv` y `trayectorias.csv`.

- **`metricas.csv`:** conteos y tiempos de distintas etapas, útiles para diagnosticar rendimiento y pérdidas.
- **`trayectorias.csv`:** ID interno, ID de ByteTrack, equipo y coordenadas por frame, útiles para revisar continuidad.
- **`Det` en la etiqueta:** confianza de YOLO en la detección de persona.
- **`ID --` y caja gris:** identidad todavía sin confirmar o en pausa.
- **`Dorsal ... OCR`:** dorsal aceptado por consenso automático.
- **`Predicho`:** caja extrapolada temporalmente sin una detección actual.
- **`Recuperados`:** cantidad de recuperaciones por galería; no mide cuántas fueron correctas.

El color del recuadro es una representación del equipo: por ejemplo, el equipo B blanco se dibuja con recuadro rojo para distinguirlo visualmente.
