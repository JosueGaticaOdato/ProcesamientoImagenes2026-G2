
import cv2
import torch
import umap
import numpy as np

from PIL import Image
from collections import defaultdict
from ultralytics import YOLO
from transformers import AutoModel, AutoProcessor

# ==========================================
# CONFIGURACION
# ==========================================

VIDEO = "partido.mp4"
SALIDA = "resultado_equipos_mejorado.mp4"

MODELO_YOLO = "yolov8s.pt"
MODELO_SIGLIP = "google/siglip-base-patch16-224"

# Confianza minima de YOLO
CONFIANZA = 0.25

# 0 = procesar el video completo
SEGUNDOS_PROCESAR = 60

# Tiempo utilizado para aprender uniformes
SEGUNDOS_APRENDIZAJE = 20

# Clasificar cada 5 frames
INTERVALO_CLASIFICACION = 5

# Cantidad maxima de muestras
MAX_MUESTRAS = 200

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

# Votos para bloquear el equipo
VOTOS_BLOQUEO = 3

# Mostrar la clasificacion desde el primer voto
MIN_VOTOS_MOSTRAR = 1

# Colores de los recuadros (BGR)
COLORES = [
    (255, 0, 0),  # Equipo A: azul
    (0, 0, 255)   # Equipo B: rojo
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# CARGAR MODELOS
# ==========================================

print("Dispositivo:", DEVICE)
print("Cargando YOLOv8...")

yolo = YOLO(MODELO_YOLO)

print("Cargando SigLIP...")

processor = AutoProcessor.from_pretrained(
    MODELO_SIGLIP,
    use_fast=False
)

siglip = AutoModel.from_pretrained(MODELO_SIGLIP)
siglip = siglip.to(DEVICE).eval()

print("Modelos cargados correctamente.")


# ==========================================
# FILTRAR PERSONAS
# ==========================================

def posible_jugador(box, frame):
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

    camisetas = []
    histogramas = []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limite = int(fps * SEGUNDOS_APRENDIZAJE)

    if total > 0:
        limite = min(limite, total)

    salto = max(1, int(fps / 2))

    for numero in range(0, limite, salto):

        cap.set(cv2.CAP_PROP_POS_FRAMES, numero)

        ret, frame = cap.read()

        if not ret:
            break

        resultados = yolo.predict(
            frame,
            classes=[0],
            conf=CONFIANZA,
            verbose=False,
            imgsz=960
        )

        boxes = resultados[0].boxes

        if boxes is None:
            continue

        for box in boxes.xyxy.cpu().numpy():

            if not posible_jugador(box, frame):
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

    return (
        reductor,
        centros_equipos,
        centro_otro,
        centros_siglip,
        radios,
        histogramas_equipos
    )


# ==========================================
# CLASIFICAR CAMISETA
# ==========================================

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
    similitud_hsv
):
    """
    Dibuja el recuadro y muestra:
    equipo, ID, confianza YOLO,
    similitud SigLIP y HSV.
    """

    x1, y1, x2, y2 = map(int, box)

    color = COLORES[equipo]

    texto_yolo = (
        f"Equipo {'A' if equipo == 0 else 'B'} "
        f"| ID {jugador_id} "
        f"| {confianza_yolo * 100:.1f}%"
    )

    texto_similitud = (
        f"SigLIP: {similitud_siglip * 100:.1f}% "
        f"| HSV: {similitud_hsv * 100:.1f}%"
    )

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

    # ======================================
    # APRENDER EQUIPOS
    # ======================================

    try:
        (
            reductor,
            centros,
            otro,
            centros_siglip,
            radios,
            histogramas_equipos
        ) = aprender_equipos(cap, fps)

    except RuntimeError as error:
        print("Error:", error)
        cap.release()
        return

    # Volver al comienzo del video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

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

    # Votos por equipo de cada ID
    historial = defaultdict(lambda: [0, 0])

    # Equipos que ya quedaron bloqueados
    equipos_bloqueados = {}

    # Guardar las ultimas similitudes
    similitudes_guardadas = defaultdict(dict)

    # Ultimo frame clasificado por jugador
    ultima_clasificacion = {}

    # Ultimo frame donde aparecio el ID
    ultima_aparicion = {}

    contador = 0

    limite = (
        int(fps * SEGUNDOS_PROCESAR)
        if SEGUNDOS_PROCESAR > 0
        else float("inf")
    )

    print("\nProcesando video...")

    while cap.isOpened() and contador < limite:

        ret, frame = cap.read()

        if not ret:
            break

        # YOLO detecta y ByteTrack sigue personas
        resultados = yolo.track(
            frame,
            persist=True,
            classes=[0],
            conf=CONFIANZA,
            imgsz=960,
            tracker="bytetrack.yaml",
            verbose=False
        )

        imagen = frame.copy()
        boxes = resultados[0].boxes

        cantidad_yolo = (
            0 if boxes is None else len(boxes)
        )

        cantidad_clasificados = 0
        cantidad_bloqueados = 0

        if boxes is not None and boxes.id is not None:

            coordenadas = boxes.xyxy.cpu().numpy()
            ids = boxes.id.int().cpu().tolist()
            confianzas = boxes.conf.cpu().numpy()

            pendientes = []

            # ==================================
            # PREPARAR CAMISETAS
            # ==================================

            for box, jugador_id in zip(
                coordenadas, ids
            ):

                if not posible_jugador(box, frame):
                    continue

                ultima_aparicion[jugador_id] = contador

                # Si ya tiene equipo bloqueado,
                # no es necesario reclasificar
                if jugador_id in equipos_bloqueados:
                    continue

                ultimo = ultima_clasificacion.get(
                    jugador_id,
                    -INTERVALO_CLASIFICACION
                )

                if contador - ultimo < INTERVALO_CLASIFICACION:
                    continue

                camiseta = recortar_camiseta(
                    frame, box
                )

                if camiseta is None:
                    continue

                histograma = obtener_histograma_hsv(
                    camiseta
                )

                if histograma is None:
                    continue

                pendientes.append(
                    (jugador_id, camiseta, histograma)
                )

                ultima_clasificacion[jugador_id] = contador

            # ==================================
            # CLASIFICAR CON SIGLIP + UMAP + HSV
            # ==================================

            for inicio in range(
                0, len(pendientes), 8
            ):

                lote = pendientes[inicio:inicio + 8]

                imagenes = [
                    camiseta
                    for _, camiseta, _ in lote
                ]

                vectores = obtener_caracteristicas(
                    imagenes
                )

                # Reducir una vez por lote
                reducidos = reductor.transform(
                    vectores
                )

                for (
                    jugador_id,
                    camiseta,
                    histograma
                ), vector, reducido in zip(
                    lote,
                    vectores,
                    reducidos
                ):

                    equipo, sim_siglip, sim_hsv = (
                        clasificar_camiseta(
                            vector,
                            reducido,
                            histograma,
                            centros,
                            otro,
                            centros_siglip,
                            radios,
                            histogramas_equipos
                        )
                    )

                    if equipo is None:
                        continue

                    # Guardar voto
                    historial[jugador_id][equipo] += 1

                    # Guardar similitudes de ese equipo
                    similitudes_guardadas[
                        jugador_id
                    ][equipo] = (
                        sim_siglip,
                        sim_hsv
                    )

                    votos = historial[jugador_id]

                    equipo_mayoritario = int(
                        np.argmax(votos)
                    )

                    diferencia = abs(
                        votos[0] - votos[1]
                    )

                    # Bloquear al alcanzar 3 votos
                    # de ventaja de un equipo
                    if diferencia >= VOTOS_BLOQUEO:

                        equipos_bloqueados[
                            jugador_id
                        ] = equipo_mayoritario

                        print(
                            f"ID {jugador_id} bloqueado "
                            f"en Equipo "
                            f"{'A' if equipo_mayoritario == 0 else 'B'}"
                        )

            # ==================================
            # DIBUJAR JUGADORES
            # ==================================

            for box, jugador_id, confianza in zip(
                coordenadas,
                ids,
                confianzas
            ):

                if not posible_jugador(box, frame):
                    continue

                # Si esta bloqueado, conservar equipo
                if jugador_id in equipos_bloqueados:
                    equipo = equipos_bloqueados[jugador_id]
                    cantidad_bloqueados += 1

                else:
                    # Mostrar equipo provisional
                    votos = historial.get(jugador_id)

                    if votos is None:
                        continue

                    if sum(votos) < MIN_VOTOS_MOSTRAR:
                        continue

                    # Evitar mostrar empates
                    if votos[0] == votos[1]:
                        continue

                    equipo = int(np.argmax(votos))

                similitudes = similitudes_guardadas[
                    jugador_id
                ].get(equipo)

                if similitudes is None:
                    continue

                sim_siglip, sim_hsv = similitudes

                cantidad_clasificados += 1

                dibujar_jugador(
                    imagen=imagen,
                    box=box,
                    jugador_id=jugador_id,
                    equipo=equipo,
                    confianza_yolo=float(confianza),
                    similitud_siglip=sim_siglip,
                    similitud_hsv=sim_hsv
                )

        # ==================================
        # DIAGNOSTICO DE DETECCIONES
        # ==================================

        if contador % 30 == 0:
            print(
                f"Frame {contador} | "
                f"YOLO: {cantidad_yolo} | "
                f"Clasificados: {cantidad_clasificados} | "
                f"Bloqueados: {cantidad_bloqueados}"
            )

        # ==================================
        # LIMPIAR IDS ANTIGUOS
        # ==================================

        if contador % 300 == 0:

            antiguos = [
                jugador_id
                for jugador_id, ultimo in ultima_aparicion.items()
                if contador - ultimo > fps * 15
            ]

            for jugador_id in antiguos:
                historial.pop(jugador_id, None)
                equipos_bloqueados.pop(jugador_id, None)
                similitudes_guardadas.pop(jugador_id, None)
                ultima_clasificacion.pop(jugador_id, None)
                ultima_aparicion.pop(jugador_id, None)

        # ==================================
        # GUARDAR Y MOSTRAR VIDEO
        # ==================================

        writer.write(imagen)
        contador += 1

        if contador % 100 == 0:
            print(
                f"Frames procesados: {contador} "
                f"| Segundos: {contador / fps:.1f}"
            )

        escala = min(1.0, 1000 / ancho)

        vista = cv2.resize(
            imagen,
            None,
            fx=escala,
            fy=escala
        )

        cv2.imshow(
            "YOLOv8 + SigLIP + UMAP + HSV",
            vista
        )

        # Salir con Q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print("\nProcesamiento finalizado.")
    print("Video generado:", SALIDA)


# ==========================================
# INICIAR PROGRAMA
# ==========================================

if __name__ == "__main__":
    procesar_video()
