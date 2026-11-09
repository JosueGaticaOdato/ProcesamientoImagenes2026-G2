
import cv2
import argparse
import csv
import torch
import numpy as np
from time import perf_counter

from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from seguimiento import SeguimientoJugadores
from apariencia import LectorDorsal, recortes_persona, descriptor_rapido
from apariencia import obtener_contorno_cancha
from apariencia import clasificar_uniforme_azul_blanco
from apariencia import pies_sobre_piso
from movimiento_camara import MovimientoCamara
from reid_ligero import Reidentificador, RUTA_PESOS
from dorsales import ControlDorsales

# ==========================================
# CONFIGURACION
# ==========================================

VIDEO = "video_recortado.mp4"
SEGUNDO_INICIO = 23
SALIDA = "resultado_equipos_mejorado.mp4"
# False permite procesar con OpenCV sin soporte de ventanas (headless).
MOSTRAR_VISTA_PREVIA = False
FILTRAR_PISO_CANCHA = True
# Calibrado sobre los uniformes del video_recortado: A azul, B blanco.
UNIFORMES_AZUL_BLANCO = True

# Las regiones de apariencia y OCR se extraen del frame HD original.
YOLO_IMGSZ = 640
USAR_OCR_DORSALES = False
INTERVALO_OCR = 30
MAX_OCR_POR_TURNO = 2
INTERVALO_SIGLIP = 15
USAR_SIGLIP = False
USAR_REID = RUTA_PESOS.is_file()
INTERVALO_REID = 10

MODELO_YOLO = "yolov8s.pt"
MODELO_SIGLIP = "google/siglip-base-patch16-224"

# Confianza minima de YOLO
CONFIANZA = 0.25

# Conservar detecciones debiles para la segunda asociacion de ByteTrack.
CONFIANZA_TRACKING = 0.10
TRACKER = str(Path(__file__).with_name("bytetrack_jugadores.yaml"))

# 0 = procesar el video completo
SEGUNDOS_PROCESAR = 12

# Tiempo utilizado para aprender uniformes
SEGUNDOS_APRENDIZAJE = 20

# Cantidad maxima de muestras
MAX_MUESTRAS = 80

# Configuracion UMAP
UMAP_COMPONENTES = 3
UMAP_VECINOS = 15
UMAP_DISTANCIA = 0.0

# Configuracion HSV
HSV_BINS = (18, 8)
SIMILITUD_HSV_MINIMA = 0.40
MARGEN_HSV = 0.02

# Parametros de SigLIP
SIMILITUD_MINIMA = 0.50
MARGEN_SIGLIP = 0.01

# Distancia maxima en UMAP
FACTOR_DISTANCIA = 2.5

# No mostrar un equipo provisional: el color aparece al confirmar el equipo.
COLOR_SIN_EQUIPO = (160, 160, 160)

# Colores de los recuadros (BGR)
COLORES = [
    (255, 0, 0),  # Equipo A: azul
    (0, 0, 255),  # Equipo B: rojo
    (0, 220, 220) # Posibles arbitros: amarillo
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Evitar competencia entre los pools de OpenCV y PyTorch durante la inferencia.
cv2.setNumThreads(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tracking desde un segmento del video')
    parser.add_argument('--inicio', type=float, default=SEGUNDO_INICIO)
    parser.add_argument('--segundos', type=float, default=SEGUNDOS_PROCESAR,
                        help='Duracion del segmento; 0 procesa hasta el final')
    parser.add_argument('--sin-ocr', action='store_true', help='Prueba rapida sin lectura de dorsales')
    parser.add_argument('--verificar-dorsales', action='store_true', help='Validar identidad con consenso OCR, con presupuesto limitado')
    parser.add_argument('--siglip', action='store_true', help='Apariencia profunda; mucho mas lento en CPU')
    parser.add_argument('--sin-reid', action='store_true', help='Solo color y textura, sin OSNet')
    parser.add_argument('--salida', default='diagnostico_tracking.mp4')
    opciones = parser.parse_args()
    if opciones.inicio < 0 or opciones.segundos < 0:
        parser.error('Inicio y duracion deben ser positivos o cero')
    if opciones.sin_ocr and opciones.verificar_dorsales:
        parser.error('--sin-ocr y --verificar-dorsales son opciones incompatibles')
    SEGUNDO_INICIO = opciones.inicio
    SEGUNDOS_PROCESAR = opciones.segundos
    USAR_OCR_DORSALES = (USAR_OCR_DORSALES or opciones.verificar_dorsales) and not opciones.sin_ocr
    SALIDA = opciones.salida
    USAR_SIGLIP = opciones.siglip
    USAR_REID = USAR_REID and not opciones.sin_reid and not USAR_SIGLIP

# El agrupamiento generico requiere las caracteristicas profundas.
USAR_SIGLIP = USAR_SIGLIP or not UNIFORMES_AZUL_BLANCO

# ==========================================
# CARGAR MODELOS
# ==========================================

print("Dispositivo:", DEVICE)
print("Cargando YOLOv8...")

yolo = YOLO(MODELO_YOLO)

if USAR_SIGLIP:
    from transformers import AutoModel, AutoProcessor
    print("Cargando SigLIP...")
    processor = AutoProcessor.from_pretrained(MODELO_SIGLIP, use_fast=False)
    siglip = AutoModel.from_pretrained(MODELO_SIGLIP).to(DEVICE).eval()
else:
    print('SigLIP desactivado. Apariencia: OSNet ligero + color/textura.' if USAR_REID
          else 'Modo rapido: apariencia por color y textura; SigLIP desactivado.')

print("Modelos cargados correctamente.")


# ==========================================
# FILTRAR PERSONAS
# ==========================================

def contorno_cancha(frame):
    """Delimitar parquet en tomas abiertas; devolver None en otros planos."""
    return obtener_contorno_cancha(frame) if FILTRAR_PISO_CANCHA else None


def posible_jugador(box, frame, cancha=None):
    """
    Filtra personas muy pequeñas o alejadas
    de la zona aproximada de juego.
    """

    x1, y1, x2, y2 = map(float, box)

    alto, ancho = frame.shape[:2]

    altura_persona = y2 - y1
    centro_x = (x1 + x2) / 2

    if altura_persona < 45:
        return False

    if centro_x < ancho * 0.02:
        return False

    if centro_x > ancho * 0.98:
        return False

    if y2 < alto * 0.35:
        return False

    if cancha is not None and cv2.pointPolygonTest(cancha, (centro_x, y2), True) < ancho * .006:
        return False
    if cancha is not None and not pies_sobre_piso(frame, box):
        # Los arbitros pueden estar en el pasillo junto a la linea lateral.
        camiseta = recortar_camiseta(frame, box)
        uniforme = (clasificar_uniforme_azul_blanco(camiseta, devolver_confianza=True)
                    if UNIFORMES_AZUL_BLANCO and camiseta is not None else (None, False))
        if uniforme != (2, True):
            return False

    return True


# ==========================================
# RECORTAR CAMISETA
# ==========================================

def recortar_camiseta(frame, box):
    """
    Obtiene la zona del torso de una persona
    para analizar su camiseta.
    """

    x1, y1, x2, y2 = map(int, box)

    ancho = x2 - x1
    alto = y2 - y1

    if ancho < 15 or alto < 45:
        return None

    xa = max(0, int(x1 + ancho * 0.20))
    xb = min(frame.shape[1], int(x1 + ancho * 0.80))

    ya = max(0, int(y1 + alto * 0.18))
    yb = min(frame.shape[0], int(y1 + alto * 0.55))

    recorte = frame[ya:yb, xa:xb]

    if recorte.size == 0:
        return None

    rgb = cv2.cvtColor(
        recorte,
        cv2.COLOR_BGR2RGB
    )

    return Image.fromarray(rgb)


# ==========================================
# OBTENER HISTOGRAMA HSV
# ==========================================

def obtener_histograma_hsv(camiseta):
    """
    Obtiene la distribucion de colores
    de la camiseta en formato HSV.
    """

    if camiseta is None:
        return None

    rgb = np.array(camiseta)

    hsv = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2HSV
    )

    histograma = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        list(HSV_BINS),
        [0, 180, 0, 256]
    )

    histograma = histograma.astype(np.float32)

    # Normalizar para que sume 1
    histograma /= max(
        float(np.sum(histograma)),
        1e-8
    )

    return histograma.flatten()


# ==========================================
# EXTRAER CARACTERISTICAS SIGLIP
# ==========================================

def obtener_caracteristicas(imagenes):
    """
    Convierte camisetas en vectores SigLIP
    para comparar su similitud visual.
    """

    if not imagenes:
        return np.empty((0, 0), dtype=np.float32)

    entradas = processor(
        images=imagenes,
        return_tensors="pt"
    )

    pixeles = entradas["pixel_values"].to(DEVICE)

    with torch.inference_mode():

        salida = siglip.get_image_features(
            pixel_values=pixeles
        )

        if hasattr(salida, "pooler_output"):
            vectores = salida.pooler_output
        elif isinstance(salida, tuple):
            vectores = salida[0]
        else:
            vectores = salida

        vectores = torch.nn.functional.normalize(
            vectores,
            dim=-1
        )

    return vectores.cpu().numpy().astype(np.float32)


# ==========================================
# APRENDER EQUIPOS
# ==========================================

def aprender_equipos(cap, fps):
    """
    Aprende uniformes usando:
    SigLIP + UMAP + K-means + HSV.
    """

    print("\nAprendiendo uniformes...")

    archivo_video = Path(VIDEO)
    datos_video = archivo_video.stat()
    firma = repr((str(archivo_video.resolve()), datos_video.st_size,
                  datos_video.st_mtime_ns, float(SEGUNDO_INICIO), SEGUNDOS_APRENDIZAJE,
                  MAX_MUESTRAS, YOLO_IMGSZ, MODELO_SIGLIP, FILTRAR_PISO_CANCHA, 'v1'))
    archivo_cache = Path(__file__).with_name('uniformes_cache.npz')
    if archivo_cache.exists():
        try:
            with np.load(archivo_cache, allow_pickle=False) as cache:
                if str(cache['firma'].item()) == firma:
                    print('Uniformes recuperados del cache; se omite el aprendizaje.')
                    return (None, None, None, cache['centros'].copy(), None,
                            cache['histogramas'].copy(), cache['otro'].copy())
        except (OSError, ValueError, KeyError):
            print('Cache de uniformes invalido; se vuelve a aprender.')

    camisetas = []
    histogramas = []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    inicio = int(fps * SEGUNDO_INICIO)
    limite = inicio + int(fps * SEGUNDOS_APRENDIZAJE)

    if total > 0:
        limite = min(limite, total)

    salto = max(1, int(fps / 2))

    for numero in range(inicio, limite, salto):

        cap.set(cv2.CAP_PROP_POS_FRAMES, numero)

        ret, frame = cap.read()

        if not ret:
            break

        cancha = contorno_cancha(frame)
        resultados = yolo.predict(
            frame,
            classes=[0],
            conf=CONFIANZA,
            verbose=False,
            imgsz=YOLO_IMGSZ
        )

        boxes = resultados[0].boxes

        if boxes is None:
            continue

        for box in boxes.xyxy.cpu().numpy():

            if not posible_jugador(box, frame, cancha):
                continue

            camiseta = recortar_camiseta(frame, box)

            if camiseta is None:
                continue

            histograma = obtener_histograma_hsv(
                camiseta
            )

            if histograma is None:
                continue

            camisetas.append(camiseta)
            histogramas.append(histograma)

            if len(camisetas) >= MAX_MUESTRAS:
                break

        if len(camisetas) >= MAX_MUESTRAS:
            break

    print("Camisetas encontradas:", len(camisetas))

    if len(camisetas) < 10:
        raise RuntimeError(
            "No hay suficientes camisetas para aprender."
        )

    # ======================================
    # EXTRAER CARACTERISTICAS SIGLIP
    # ======================================

    print("Extrayendo caracteristicas SigLIP...")

    caracteristicas = []

    for inicio in range(0, len(camisetas), 8):

        lote = camisetas[inicio:inicio + 8]

        vectores = obtener_caracteristicas(lote)

        caracteristicas.append(vectores)

    datos = np.vstack(caracteristicas)

    histogramas = np.array(
        histogramas,
        dtype=np.float32
    )

    print("Dimensiones SigLIP:", datos.shape)

    # ======================================
    # REDUCIR DIMENSIONES CON UMAP
    # ======================================

    print("Aplicando UMAP...")
    import umap

    reductor = umap.UMAP(
        n_components=UMAP_COMPONENTES,
        n_neighbors=min(
            UMAP_VECINOS,
            len(datos) - 1
        ),
        min_dist=UMAP_DISTANCIA,
        metric="cosine",
        random_state=42
    )

    datos_reducidos = reductor.fit_transform(
        datos
    ).astype(np.float32)

    print("Dimensiones UMAP:", datos_reducidos.shape)

    # ======================================
    # AGRUPAR CON K-MEANS
    # ======================================

    print("Agrupando uniformes...")

    criterios = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.001
    )

    _, etiquetas, centros_umap = cv2.kmeans(
        datos_reducidos,
        3,
        None,
        criterios,
        15,
        cv2.KMEANS_PP_CENTERS
    )

    etiquetas = etiquetas.flatten()

    cantidades = np.bincount(
        etiquetas,
        minlength=3
    )

    print("Muestras por grupo:", cantidades)

    orden = np.argsort(cantidades)[::-1]

    # Los dos grupos mayores son posibles equipos
    indices_equipos = orden[:2]
    indice_otro = orden[2]

    centros_equipos = centros_umap[indices_equipos]
    centro_otro = centros_umap[indice_otro]

    # ======================================
    # CENTROS ORIGINALES DE SIGLIP
    # ======================================

    centros_siglip = []

    for grupo in indices_equipos:

        vectores = datos[etiquetas == grupo]

        centro = np.mean(vectores, axis=0)

        centro /= max(
            np.linalg.norm(centro),
            1e-8
        )

        centros_siglip.append(centro)

    centros_siglip = np.array(
        centros_siglip,
        dtype=np.float32
    )

    # ======================================
    # HISTOGRAMAS HSV DE LOS EQUIPOS
    # ======================================

    histogramas_equipos = []

    for grupo in indices_equipos:

        muestras_equipo = histogramas[
            etiquetas == grupo
        ]

        histograma_promedio = np.mean(
            muestras_equipo,
            axis=0
        )

        histograma_promedio /= max(
            float(np.sum(histograma_promedio)),
            1e-8
        )

        histogramas_equipos.append(
            histograma_promedio
        )

    histogramas_equipos = np.array(
        histogramas_equipos,
        dtype=np.float32
    )

    # ======================================
    # RADIOS DE LOS GRUPOS UMAP
    # ======================================

    radios = []

    for grupo in indices_equipos:

        puntos = datos_reducidos[
            etiquetas == grupo
        ]

        distancias = np.linalg.norm(
            puntos - centros_umap[grupo],
            axis=1
        )

        radio = float(
            np.percentile(distancias, 90)
        )

        radios.append(max(radio, 0.1))

    radios = np.array(
        radios,
        dtype=np.float32
    )

    print("\nEquipos aprendidos:")
    print("Equipo A:", indices_equipos[0])
    print("Equipo B:", indices_equipos[1])
    print("Posible grupo ajeno:", indice_otro)
    otro_siglip = np.mean(datos[etiquetas == indice_otro], axis=0)
    np.savez_compressed(archivo_cache, firma=np.array(firma), centros=centros_siglip,
                        histogramas=histogramas_equipos, otro=otro_siglip)

    return (
        reductor,
        centros_equipos,
        centro_otro,
        centros_siglip,
        radios,
        histogramas_equipos,
        otro_siglip
    )


# ==========================================
# CLASIFICAR CAMISETA
# ==========================================

def clasificar_rapido(vector, histograma, centros_siglip, histogramas_equipos, otro_siglip):
    """Inferencia directa: UMAP solo se usa durante el aprendizaje inicial."""
    similitudes = centros_siglip @ vector
    equipo = int(np.argmax(similitudes))
    sim = float(similitudes[equipo])
    colores = [float(np.minimum(histograma, h).sum()) for h in histogramas_equipos]
    otro_normalizado = otro_siglip / max(float(np.linalg.norm(otro_siglip)), 1e-8)
    if float(otro_normalizado @ vector) > sim + .02:
        return 2, sim, colores[equipo]
    if sim < SIMILITUD_MINIMA or sim - float(similitudes[1 - equipo]) < MARGEN_SIGLIP:
        return None, sim, colores[equipo]
    if colores[1 - equipo] > colores[equipo] + .15:
        return None, sim, colores[equipo]
    return equipo, sim, colores[equipo]

def clasificar_camiseta(
    vector,
    reducido,
    histograma,
    centros,
    otro,
    centros_siglip,
    radios,
    histogramas_equipos
):
    """
    Clasifica con UMAP y SigLIP.
    HSV se utiliza como apoyo para detectar
    contradicciones claras de color.
    """

    # ======================================
    # UMAP: EQUIPO MAS CERCANO
    # ======================================

    distancias = np.linalg.norm(
        centros - reducido,
        axis=1
    )

    equipo_umap = int(np.argmin(distancias))

    distancia_equipo = float(
        distancias[equipo_umap]
    )

    distancia_otro = float(
        np.linalg.norm(otro - reducido)
    )

    # ======================================
    # SIGLIP: SIMILITUD DE CAMISETA
    # ======================================

    similitudes = centros_siglip @ vector

    equipo_siglip = int(np.argmax(similitudes))

    similitud_siglip = float(
        similitudes[equipo_siglip]
    )

    # Rechazar similitudes demasiado bajas
    if similitud_siglip < SIMILITUD_MINIMA:
        return None, similitud_siglip, None

    # UMAP y SigLIP deben coincidir
    if equipo_siglip != equipo_umap:
        return None, similitud_siglip, None

    # Diferencia entre los dos equipos SigLIP
    diferencia_siglip = float(
        similitudes[equipo_siglip]
        - similitudes[1 - equipo_siglip]
    )

    if diferencia_siglip < MARGEN_SIGLIP:
        return None, similitud_siglip, None

    # Descartar solo si UMAP esta muy lejos
    if distancia_equipo > (
        radios[equipo_umap] * FACTOR_DISTANCIA
    ):
        return None, similitud_siglip, None

    # Descartar si el tercer grupo es
    # claramente mas cercano
    if distancia_otro + 0.2 < distancia_equipo:
        return None, similitud_siglip, None

    # ======================================
    # HSV: COMPARACION DE COLOR
    # ======================================

    similitudes_hsv = []

    for referencia in histogramas_equipos:

        similitud = cv2.compareHist(
            histograma,
            referencia,
            cv2.HISTCMP_INTERSECT
        )

        similitudes_hsv.append(float(similitud))

    equipo_hsv = int(np.argmax(similitudes_hsv))

    # Similitud HSV con el equipo seleccionado
    similitud_hsv = float(
        similitudes_hsv[equipo_siglip]
    )

    # Si HSV discrepa, solo rechazamos cuando
    # el otro equipo tiene una ventaja muy clara
    if equipo_hsv != equipo_siglip:

        diferencia_hsv = (
            similitudes_hsv[equipo_hsv]
            - similitudes_hsv[equipo_siglip]
        )

        if (
            similitudes_hsv[equipo_hsv] >= 0.70
            and diferencia_hsv >= 0.20
        ):
            return (
                None,
                similitud_siglip,
                similitud_hsv
            )

    # HSV ahora es apoyo, no un filtro obligatorio
    return (
        equipo_siglip,
        similitud_siglip,
        similitud_hsv
    )


# ==========================================
# DIBUJAR JUGADOR
# ==========================================

def dibujar_jugador(
    imagen,
    box,
    jugador_id,
    equipo,
    confianza_yolo,
    similitud_siglip,
    similitud_hsv,
    dorsal=None,
    estado=None
):
    """
    Dibuja el recuadro y muestra:
    equipo, ID, confianza YOLO,
    similitud SigLIP y HSV.
    """

    x1, y1, x2, y2 = map(int, box)

    color = COLOR_SIN_EQUIPO if equipo is None else COLORES[equipo]
    nombre_equipo = (
        "Sin identificar" if equipo is None
        else ("Posible arbitro" if equipo == 2 else f"Equipo {'A' if equipo == 0 else 'B'}")
    )

    texto_yolo = (
        f"{nombre_equipo} "
        f"| ID {jugador_id if jugador_id is not None else '--'} "
        f"| Det {confianza_yolo * 100:.1f}%"
    )
    if dorsal is not None:
        texto_yolo += f" | Dorsal {dorsal} OCR"
    if estado is not None:
        texto_yolo = f"{nombre_equipo} | ID {jugador_id if jugador_id is not None else '--'} | {estado}"

    texto_similitud = "Asociacion de identidad pendiente" if equipo is None else (
        f"SigLIP: {similitud_siglip * 100:.1f}% "
        f"| HSV: {similitud_hsv * 100:.1f}%"
    )
    if equipo is not None and UNIFORMES_AZUL_BLANCO:
        texto_similitud = 'Asociacion por apariencia' if dorsal is None else 'Referencia por consenso OCR'
    if estado is not None:
        texto_similitud = 'Identidad en pausa' if equipo is None else 'Prediccion temporal, sin deteccion actual'

    # Dibujar recuadro
    cv2.rectangle(
        imagen,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    fuente = cv2.FONT_HERSHEY_SIMPLEX
    escala = 0.45
    grosor = 1

    (w1, h1), _ = cv2.getTextSize(
        texto_yolo,
        fuente,
        escala,
        grosor
    )

    (w2, h2), _ = cv2.getTextSize(
        texto_similitud,
        fuente,
        escala,
        grosor
    )

    ancho_texto = max(w1, w2) + 10
    alto_texto = h1 + h2 + 20

    # Mantener la etiqueta dentro de la imagen
    pos_x = max(
        0,
        min(x1, imagen.shape[1] - ancho_texto)
    )

    if y1 >= alto_texto + 4:
        pos_y = y1 - alto_texto - 4
    else:
        pos_y = min(
            imagen.shape[0] - alto_texto,
            y2 + 4
        )

    # Fondo de la etiqueta
    cv2.rectangle(
        imagen,
        (pos_x, pos_y),
        (
            pos_x + ancho_texto,
            pos_y + alto_texto
        ),
        color,
        -1
    )

    # Primera linea
    cv2.putText(
        imagen,
        texto_yolo,
        (pos_x + 5, pos_y + h1 + 5),
        fuente,
        escala,
        (255, 255, 255),
        grosor,
        cv2.LINE_AA
    )

    # Segunda linea
    cv2.putText(
        imagen,
        texto_similitud,
        (pos_x + 5, pos_y + h1 + h2 + 12),
        fuente,
        escala,
        (255, 255, 255),
        grosor,
        cv2.LINE_AA
    )


# ==========================================
# PROCESAR VIDEO
# ==========================================

def procesar_video():

    cap = cv2.VideoCapture(VIDEO)

    if not cap.isOpened():
        print("No se pudo abrir el video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_inicio = int(fps * SEGUNDO_INICIO)
    total_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_video > 0 and frame_inicio >= total_video:
        print("El segundo de inicio queda fuera del video.")
        cap.release()
        return

    # ======================================
    # APRENDER EQUIPOS
    # ======================================

    try:
        if UNIFORMES_AZUL_BLANCO:
            print('Uniformes configurados para este video: A azul, B blanco, posibles arbitros gris.')
            print('Se omite el agrupamiento automatico de uniformes.')
        else:
            (
                reductor,
                centros,
                otro,
                centros_siglip,
                radios,
                histogramas_equipos,
                otro_siglip
            ) = aprender_equipos(cap, fps)

    except RuntimeError as error:
        print("Error:", error)
        cap.release()
        return

    # Aprendizaje y salida comienzan en el segmento solicitado.
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_inicio)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        SALIDA,
        fourcc,
        fps,
        (ancho, alto)
    )

    if not writer.isOpened():
        print("No se pudo crear el video.")
        cap.release()
        return

    # ======================================
    # HISTORIALES
    # ======================================

    seguimiento = SeguimientoJugadores(ventana=10, memoria=max(10, int(fps * 3)))
    movimiento_camara = MovimientoCamara()
    lector_dorsal = LectorDorsal(gpu=DEVICE == 'cuda', activo=USAR_OCR_DORSALES)
    control_dorsales = ControlDorsales(max_edad=max(30,int(fps*5)), separacion=max(1,int(fps/10)))
    cache_apariencia = {}
    cache_reid = {}
    reidentificador = Reidentificador(DEVICE) if USAR_REID else None
    print('Reidentificacion OSNet x0.25:', 'activa' if reidentificador else 'desactivada')
    diagnostico = []
    trayectorias = []
    carpeta_diagnostico = Path(SALIDA).with_suffix('')
    carpeta_diagnostico.mkdir(parents=True, exist_ok=True)

    contador = 0
    mostrar_vista_previa = MOSTRAR_VISTA_PREVIA
    ventana_abierta = False
    plano_cercano_anterior = False

    limite = (
        int(fps * SEGUNDOS_PROCESAR)
        if SEGUNDOS_PROCESAR > 0
        else float("inf")
    )

    print(f"\nProcesando desde {SEGUNDO_INICIO:.1f} s del video original...")
    inicio_procesamiento = perf_counter()
    total_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    disponibles = max(0, total_video - frame_inicio)
    total_objetivo = min(disponibles, int(limite)) if limite != float('inf') else disponibles

    while cap.isOpened() and contador < limite:

        ret, frame = cap.read()

        if not ret:
            break

        # No filtrar detecciones debiles antes de la asociacion de ByteTrack.
        inicio_detector = perf_counter()
        resultados = yolo.track(
            frame,
            persist=True,
            classes=[0],
            conf=CONFIANZA_TRACKING,
            imgsz=YOLO_IMGSZ,
            tracker=TRACKER,
            verbose=False
        )

        imagen = frame.copy()
        inicio_camara = perf_counter()
        boxes = resultados[0].boxes
        plano_cercano = False
        if boxes is not None and len(boxes):
            coords_plano = boxes.xyxy.cpu().numpy()
            areas = (coords_plano[:, 2] - coords_plano[:, 0]) * (coords_plano[:, 3] - coords_plano[:, 1])
            alturas = coords_plano[:, 3] - coords_plano[:, 1]
            plano_cercano = bool(np.any((areas > ancho * alto * .18) & (alturas > alto * .65)))
        if plano_cercano != plano_cercano_anterior:
            seguimiento.cambio_de_plano()
            movimiento_camara = MovimientoCamara()
        plano_cercano_anterior = plano_cercano
        cancha = contorno_cancha(frame)
        matriz_camara = movimiento_camara.actualizar(frame)
        if not plano_cercano:
            seguimiento.compensar_camara(matriz_camara)
        inicio_apariencia = perf_counter()
        boxes = resultados[0].boxes

        cantidad_yolo = (
            0 if boxes is None else len(boxes)
        )

        cantidad_clasificados = 0
        cantidad_bloqueados = 0

        detecciones = []
        recuadros_pendientes = []
        duplicados_descartados = 0
        ids_ocr = set()
        if plano_cercano and boxes is not None:
            recuadros_pendientes = list(zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()))
        if boxes is not None and not plano_cercano:
            pendientes = []
            ids_raw = (boxes.id.int().cpu().tolist() if boxes.id is not None
                       else [-(i + 1) for i in range(len(boxes))])
            coordenadas = boxes.xyxy.cpu().numpy()
            confianzas = boxes.conf.cpu().numpy()
            equipos_nms = None
            if UNIFORMES_AZUL_BLANCO:
                camisetas_nms = [recortar_camiseta(frame, box) for box in coordenadas]
                equipos_nms = [clasificar_uniforme_azul_blanco(c) if c is not None else None
                               for c in camisetas_nms]
            conservar = seguimiento.indices_sin_duplicados(coordenadas, confianzas, equipos_nms)
            duplicados_descartados = len(coordenadas) - len(conservar)
            coordenadas, confianzas = coordenadas[conservar], confianzas[conservar]
            ids_raw = [ids_raw[i] for i in conservar]
            lados = np.maximum(0, np.minimum(coordenadas[:,None,2:], coordenadas[None,:,2:]) -
                               np.maximum(coordenadas[:,None,:2], coordenadas[None,:,:2]))
            intersecciones = np.prod(lados, axis=2)
            areas = np.prod(coordenadas[:,2:] - coordenadas[:,:2], axis=1)
            solapes = intersecciones / np.maximum(areas[:,None] + areas[None,:] - intersecciones, 1e-8)
            np.fill_diagonal(solapes, 0)
            for indice, (box, raw_id, confianza) in enumerate(zip(
                coordenadas, ids_raw,
                confianzas
            )):
                conocido = seguimiento.conocido(raw_id, box, contador)
                if not conocido and not posible_jugador(box, frame, cancha):
                    continue
                camiseta = recortar_camiseta(frame, box)
                if camiseta is None:
                    recuadros_pendientes.append((box, float(confianza)))
                    continue
                histograma = obtener_histograma_hsv(camiseta)
                regiones = recortes_persona(frame, box)
                pendientes.append((box, raw_id, float(confianza), camiseta, histograma, regiones,
                                   bool(np.any(solapes[indice] > .20))))

            if reidentificador is not None:
                refrescar_reid = [p for p in pendientes if not p[6] and 'cuerpo' in p[5] and
                                  (p[1] not in cache_reid or
                                   contador - cache_reid[p[1]][0] >= INTERVALO_REID)]
                for inicio in range(0, len(refrescar_reid), 8):
                    lote_reid = refrescar_reid[inicio:inicio + 8]
                    embeddings = reidentificador.extraer([p[5]['cuerpo'] for p in lote_reid])
                    for registro, embedding in zip(lote_reid, embeddings):
                        cache_reid[registro[1]] = (contador, embedding)

            # Redes costosas solo para IDs nuevos o cada 15 frames.
            refrescar = [p for p in pendientes if USAR_SIGLIP and (p[1] < 0 or p[1] not in cache_apariencia
                         or contador - cache_apariencia[p[1]][0] >= INTERVALO_SIGLIP)]
            for inicio in range(0, len(refrescar), 8):
                lote_nuevo = refrescar[inicio:inicio + 8]
                nuevos = obtener_caracteristicas([p[3] for p in lote_nuevo])
                for registro, vector in zip(lote_nuevo, nuevos):
                    cache_apariencia[registro[1]] = (contador, vector)
            # Limitar OCR y rotar candidatos: nunca OCR por cada persona/frame.
            if lector_dorsal.reader is not None and contador % INTERVALO_OCR == 0 and pendientes:
                candidatos = [p for p in pendientes if p[1] >= 0 and not p[6] and 'dorsal' in p[5]
                              and (not UNIFORMES_AZUL_BLANCO or
                                   clasificar_uniforme_azul_blanco(p[3],True) in ((0,True),(1,True)))]
                candidatos.sort(key=lambda p:(control_dorsales.prioridad(p[1],contador),p[0][3]-p[0][1]),reverse=True)
                ids_ocr = {p[1] for p in candidatos[:MAX_OCR_POR_TURNO]}
            # Posicion, color y regiones baratas se actualizan en TODOS los frames.
            for inicio in range(0, len(pendientes), 8):
                lote = pendientes[inicio:inicio + 8]
                for registro in lote:
                    box, raw_id, confianza, camiseta, histograma, regiones, ocluido = registro
                    partes = {nombre: descriptor_rapido(recorte)
                              for nombre, recorte in regiones.items() if nombre != 'dorsal'}
                    if USAR_SIGLIP:
                        _, vector = cache_apariencia[raw_id]
                    else:
                        # Mismo descriptor y dimension durante toda la ejecucion.
                        vector = descriptor_rapido(np.asarray(camiseta)[:, :, ::-1].copy())
                    lectura_dorsal = (lector_dorsal.leer(regiones.get('dorsal'))
                                      if raw_id in ids_ocr else None)
                    if raw_id in ids_ocr:
                        control_dorsales.registrar(raw_id, lectura_dorsal, contador)
                    dorsal = (control_dorsales.confirmado(raw_id, contador, reciente=control_dorsales.separacion)
                              if not ocluido else None)
                    if UNIFORMES_AZUL_BLANCO:
                        equipo_torso = clasificar_uniforme_azul_blanco(camiseta)
                        equipo, equipo_fuerte = clasificar_uniforme_azul_blanco(
                            camiseta, devolver_confianza=True, shorts=regiones.get('shorts'))
                        uniforme_contradictorio = equipo_torso in (0, 1) and equipo is None
                        sim_siglip, sim_hsv = 0.0, 0.0
                    else:
                        equipo_fuerte = False
                        uniforme_contradictorio = False
                        equipo, sim_siglip, sim_hsv = clasificar_rapido(
                            vector, histograma, centros_siglip,
                            histogramas_equipos, otro_siglip
                        )
                    detecciones.append({
                        'box': box.copy(), 'raw_id': raw_id,
                        'confianza': confianza, 'vector': vector.copy(),
                        'histograma': histograma.copy(), 'equipo': equipo,
                        'partes': partes, 'dorsal': dorsal,
                        'dorsal_lectura': lectura_dorsal, 'dorsal_verificado': dorsal is not None,
                        'equipo_fuerte': equipo_fuerte,
                        'uniforme_contradictorio': uniforme_contradictorio,
                        'reid': cache_reid.get(raw_id, (None, None))[1],
                        'ocluido': ocluido,
                        'sim_siglip': sim_siglip or 0.0,
                        'sim_hsv': sim_hsv or 0.0,
                    })

        inicio_asociacion = perf_counter()
        visibles = seguimiento.actualizar(detecciones, contador)
        predicciones = ([] if plano_cercano else
                        seguimiento.predecir(contador, ancho, alto, visibles))
        fin_asociacion = perf_counter()
        if contador % 300 == 0:
            control_dorsales.limpiar(contador)
            cache_apariencia = {k: v for k, v in cache_apariencia.items()
                                if contador - v[0] <= max(300, fps * 15)}
            cache_reid = {k: v for k, v in cache_reid.items() if contador - v[0] <= max(300, fps * 15)}
        cantidad_clasificados = len(visibles)
        cantidad_bloqueados = sum(t['id'] is not None for t in seguimiento.tracks)
        diagnostico.append({
            'frame': contador, 'segundo_video': SEGUNDO_INICIO + contador / fps,
            'yolo': cantidad_yolo, 'aptas_para_tracking': len(detecciones),
            'confirmados_visibles': cantidad_clasificados,
            'ids_reservados': cantidad_bloqueados,
            'recuperaciones': seguimiento.recuperaciones,
            'camara_compensada': matriz_camara is not None,
            'primer_plano': plano_cercano, 'predicciones': len(predicciones),
            'duplicados_descartados': duplicados_descartados,
            'intentos_ocr': len(ids_ocr),
            'lecturas_ocr': sum(d['dorsal_lectura'] is not None for d in detecciones),
            'ids_con_dorsal': sum(seguimiento.dorsal_confirmado(t) is not None
                                  for t in seguimiento.tracks if t['id'] is not None),
            'ms_detector': (inicio_camara - inicio_detector) * 1000,
            'ms_camara': (inicio_apariencia - inicio_camara) * 1000,
            'ms_apariencia_ocr': (inicio_asociacion - inicio_apariencia) * 1000,
            'ms_asociacion': (fin_asociacion - inicio_asociacion) * 1000,
        })
        # No ocultar detecciones por no tener todavia una identidad confirmada.
        raw_confirmados = {j['raw_id'] for j in visibles}
        recuadros_pendientes.extend((d['box'], d['confianza']) for d in detecciones
                                    if d['raw_id'] not in raw_confirmados)
        for box, confianza in recuadros_pendientes:
            dibujar_jugador(imagen, box, None, None, confianza, 0.0, 0.0,
                            estado='Primer plano' if plano_cercano else None)
        for jugador in predicciones:
            dibujar_jugador(imagen, jugador['box'], jugador['jugador_id'], jugador['equipo'],
                            0.0, 0.0, 0.0, dorsal=jugador['dorsal'],
                            estado=f"Predicho {jugador['frames_predichos']}f")
        for jugador in visibles:
            trayectorias.append({
                'frame': contador, 'segundo_video': SEGUNDO_INICIO + contador / fps,
                'id': jugador['jugador_id'], 'id_bytetrack': jugador['raw_id'],
                'equipo_bloqueado': jugador['equipo'],
                'dorsal_consenso': jugador.get('dorsal') or '',
                'dorsal_lectura': jugador.get('dorsal_lectura') or '',
                'dorsal_verificado': jugador.get('dorsal_verificado',False),
                'equipo_observado': next((d['equipo'] for d in detecciones
                                          if d['raw_id'] == jugador['raw_id']), None),
                'x1': float(jugador['box'][0]), 'y1': float(jugador['box'][1]),
                'x2': float(jugador['box'][2]), 'y2': float(jugador['box'][3]),
            })
            dibujar_jugador(
                imagen, jugador['box'], jugador['jugador_id'], jugador['equipo'],
                jugador['confianza'], jugador['sim_siglip'], jugador['sim_hsv'],
                dorsal=jugador['dorsal']
            )

        if contador % 30 == 0:
            print(
                f"Frame {contador} | Video: {SEGUNDO_INICIO + contador / fps:.1f}s "
                f"| YOLO: {cantidad_yolo} | IDs asociados: {cantidad_clasificados} "
                f"| Sin confirmar: {len(recuadros_pendientes)} | IDs reservados: {cantidad_bloqueados}/12"
                f" | Recuperados: {seguimiento.recuperaciones}"
                f" | Predichos: {len(predicciones)} | Primer plano: {plano_cercano}"
                f" | Duplicados descartados: {duplicados_descartados} | OCR: {len(ids_ocr)} intentos"
            )


        # ==================================
        # GUARDAR Y MOSTRAR VIDEO
        # ==================================

        if contador % max(1, int(fps * 3)) == 0:
            cv2.imwrite(str(carpeta_diagnostico / f'frame_{contador:06d}.jpg'), imagen)
        writer.write(imagen)
        contador += 1

        if contador % 100 == 0:
            transcurrido = perf_counter() - inicio_procesamiento
            velocidad = contador / max(transcurrido, .001)
            faltantes = max(0, total_objetivo - contador)
            print(
                f"Frames procesados: {contador} "
                f"/ {total_objetivo} | {velocidad:.2f} frames/s "
                f"| Restante aprox: {faltantes / max(velocidad, .001) / 60:.1f} min"
            )

        if mostrar_vista_previa:
            escala = min(1.0, 1000 / ancho)
            vista = cv2.resize(imagen, None, fx=escala, fy=escala)
            try:
                cv2.imshow("YOLOv8 + SigLIP + UMAP + HSV", vista)
                ventana_abierta = True
                # Q solo se consulta cuando la vista previa esta disponible.
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except cv2.error as error:
                print("Vista previa no disponible; se continua guardando el video:", error)
                mostrar_vista_previa = False

    cap.release()
    writer.release()
    if contador > 0:
        cv2.imwrite(str(carpeta_diagnostico / 'ultimo_frame.jpg'), imagen)
    if diagnostico:
        with open(carpeta_diagnostico / 'metricas.csv', 'w', newline='', encoding='utf-8') as archivo:
            registro = csv.DictWriter(archivo, fieldnames=list(diagnostico[0]))
            registro.writeheader()
            registro.writerows(diagnostico)
    if trayectorias:
        with open(carpeta_diagnostico / 'trayectorias.csv', 'w', newline='', encoding='utf-8') as archivo:
            registro = csv.DictWriter(archivo, fieldnames=list(trayectorias[0]))
            registro.writeheader()
            registro.writerows(trayectorias)
    if ventana_abierta:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    print("\nProcesamiento finalizado.")
    print("Video generado:", SALIDA)


# ==========================================
# INICIAR PROGRAMA
# ==========================================

if __name__ == "__main__":
    procesar_video()
