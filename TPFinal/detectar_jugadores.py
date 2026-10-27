
import cv2
import csv
import numpy as np

from pathlib import Path
from collections import Counter
from sklearn.cluster import KMeans
from ultralytics import YOLO


# ==========================================
# CONFIGURACION
# ==========================================

VIDEO = "partido.mp4"
MODELO_YOLO = "yolov8s.pt"

CARPETA_SALIDA = Path("jugadores_detectados")

CANTIDAD_FRAMES = 3
CONFIANZA_YOLO = 0.25
YOLO_IMGSZ = 960

# True: descarta detecciones segun su posicion
# False: guarda todas las personas detectadas
FILTRAR_JUGADORES = True

# Clasificacion por color
MIN_MUESTRAS_CLUSTER = 4
MARGEN_MINIMO = 0.10

# Guardar tambien imagenes descartadas
GUARDAR_DESCARTADOS = True


# ==========================================
# FILTRO DE PERSONAS
# ==========================================

def posible_jugador(box, frame):

    x1, y1, x2, y2 = map(float, box)

    alto, ancho = frame.shape[:2]

    altura_persona = y2 - y1
    centro_x = (x1 + x2) / 2

    if altura_persona < 45:
        return False, "Altura insuficiente"

    if centro_x < ancho * 0.02:
        return False, "Extremo izquierdo"

    if centro_x > ancho * 0.98:
        return False, "Extremo derecho"

    if y2 < alto * 0.35:
        return False, "Zona superior del video"

    return True, "Valido"


# ==========================================
# RECORTAR PERSONA
# ==========================================

def recortar_persona(frame, box):

    alto, ancho = frame.shape[:2]

    x1, y1, x2, y2 = map(int, box)

    x1 = max(0, min(x1, ancho))
    x2 = max(0, min(x2, ancho))

    y1 = max(0, min(y1, alto))
    y2 = max(0, min(y2, alto))

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2].copy()


# ==========================================
# EXTRAER CAMISETA
# ==========================================

def extraer_camiseta(persona):

    """
    Recorta una region central del torso.
    No elimina completamente el fondo.
    """

    alto, ancho = persona.shape[:2]

    xa = int(ancho * 0.25)
    xb = int(ancho * 0.75)

    ya = int(alto * 0.18)
    yb = int(alto * 0.48)

    if xb <= xa or yb <= ya:
        return None

    recorte = persona[ya:yb, xa:xb]

    if recorte.size == 0:
        return None

    return recorte


# ==========================================
# CARACTERISTICAS DE COLOR
# ==========================================

def obtener_caracteristicas(camiseta):

    """
    Extrae:
    - Histograma HSV
    - Luminosidad
    - Saturacion media

    Usa los datos para agrupar uniformes.
    """

    if camiseta is None:
        return None

    hsv = cv2.cvtColor(
        camiseta,
        cv2.COLOR_BGR2HSV
    )

    # Histograma conjunto H y S
    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [12, 6],
        [0, 180, 0, 256]
    ).flatten()

    hist = hist.astype(np.float32)
    hist /= max(hist.sum(), 1e-8)

    # Distribucion de brillo
    brillo_hist = cv2.calcHist(
        [hsv],
        [2],
        None,
        [8],
        [0, 256]
    ).flatten().astype(np.float32)

    brillo_hist /= max(
        brillo_hist.sum(), 1e-8
    )

    brillo = float(np.median(
        hsv[:, :, 2]
    )) / 255.0

    saturacion = float(np.median(
        hsv[:, :, 1]
    )) / 255.0

    # Damos peso al color y la luminosidad
    caracteristicas = np.concatenate([
        hist * 2.0,
        brillo_hist * 1.5,
        np.array([
            brillo * 1.5,
            saturacion * 0.5
        ], dtype=np.float32)
    ])

    return (
        caracteristicas.astype(np.float32),
        brillo,
        saturacion
    )


# ==========================================
# CLASIFICAR LOS DOS EQUIPOS
# ==========================================

def clasificar_equipos(registros):

    """
    Agrupa las camisetas en dos conjuntos.
    Asigna el Equipo A al grupo mas claro
    y el Equipo B al mas oscuro.

    No es una identificacion oficial:
    son grupos provisionales por color.
    """

    validos = [
        r for r in registros
        if r["aceptado"]
        and r["caracteristicas"] is not None
    ]

    if len(validos) < MIN_MUESTRAS_CLUSTER:

        print(
            "No hay suficientes camisetas "
            "para aprender dos equipos."
        )

        return

    datos = np.vstack([
        r["caracteristicas"]
        for r in validos
    ])

    # Buscar dos grupos
    kmeans = KMeans(
        n_clusters=2,
        n_init=20,
        random_state=42
    )

    etiquetas = kmeans.fit_predict(datos)

    # Determinar que grupo es mas claro
    luminosidades = {}

    for grupo in (0, 1):

        valores = [
            r["brillo"]
            for r, etiqueta in zip(
                validos, etiquetas
            )
            if etiqueta == grupo
        ]

        luminosidades[grupo] = (
            float(np.mean(valores))
            if valores else 0.0
        )

    grupo_claro = max(
        luminosidades,
        key=luminosidades.get
    )

    grupo_oscuro = 1 - grupo_claro

    # Distancias a los centros de K-means
    distancias = kmeans.transform(datos)

    for i, (registro, etiqueta) in enumerate(
        zip(validos, etiquetas)
    ):

        d = distancias[i]

        distancia_propia = float(d[etiqueta])
        distancia_otra = float(d[1 - etiqueta])

        # Ventaja relativa del grupo asignado
        margen = (
            (distancia_otra - distancia_propia)
            / max(distancia_otra, 1e-8)
        )

        registro["margen"] = margen

        if margen < MARGEN_MINIMO:

            registro["equipo"] = "Sin clasificar"
            registro["motivo"] = (
                "Camiseta ambigua entre grupos"
            )

        elif etiqueta == grupo_claro:

            registro["equipo"] = "Equipo A"

        elif etiqueta == grupo_oscuro:

            registro["equipo"] = "Equipo B"


# ==========================================
# GUARDAR IMAGENES
# ==========================================

def guardar_imagenes(registros):

    carpetas = {
        "Equipo A": CARPETA_SALIDA / "Equipo_A",
        "Equipo B": CARPETA_SALIDA / "Equipo_B",
        "Sin clasificar": CARPETA_SALIDA / "Sin_clasificar",
        "Descartado": CARPETA_SALIDA / "Descartados"
    }

    for carpeta in carpetas.values():
        carpeta.mkdir(
            parents=True,
            exist_ok=True
        )

    for registro in registros:

        if (
            not registro["aceptado"]
            and not GUARDAR_DESCARTADOS
        ):
            continue

        recorte = registro["imagen"]

        if recorte is None:
            continue

        equipo = registro["equipo"]

        carpeta = carpetas[equipo]

        nombre = (
            f"frame_{registro['frame']:02d}"
            f"_persona_{registro['deteccion']:02d}.jpg"
        )

        ruta = carpeta / nombre

        if cv2.imwrite(str(ruta), recorte):

            registro["archivo"] = str(ruta)

        else:

            registro["archivo"] = "Error al guardar"


# ==========================================
# GENERAR DIAGNOSTICO TXT
# ==========================================

def generar_txt(registros, frames_procesados):

    ruta = CARPETA_SALIDA / "diagnostico.txt"

    with open(ruta, "w", encoding="utf-8") as f:

        def escribir(texto=""):
            f.write(str(texto) + "\n")

        escribir("DIAGNOSTICO DE JUGADORES - BASQUET")
        escribir("=" * 65)

        escribir(f"Video: {VIDEO}")
        escribir(f"Modelo: {MODELO_YOLO}")
        escribir(f"Frames procesados: {frames_procesados}")
        escribir(f"Confianza minima YOLO: {CONFIANZA_YOLO}")
        escribir(f"Resolucion YOLO: {YOLO_IMGSZ}")
        escribir(f"Filtro activado: {FILTRAR_JUGADORES}")

        escribir()
        escribir("IMPORTANTE")
        escribir("-" * 65)
        escribir(
            "Los equipos son grupos estimados por color."
        )
        escribir(
            "Equipo A = grupo con mayor luminosidad."
        )
        escribir(
            "Equipo B = grupo con menor luminosidad."
        )
        escribir(
            "Las cantidades corresponden a detecciones,"
        )
        escribir(
            "no a jugadores unicos."
        )
        escribir(
            "Arbitros y espectadores pueden clasificarse"
        )
        escribir(
            "erroneamente como jugadores."
        )

        # ==================================
        # RESUMEN GENERAL
        # ==================================

        conteo = Counter(
            r["equipo"]
            for r in registros
        )

        escribir()
        escribir("RESUMEN GENERAL")
        escribir("-" * 65)

        escribir(f"Personas detectadas: {len(registros)}")
        escribir(
            f"Detecciones validas: "
            f"{sum(r['aceptado'] for r in registros)}"
        )

        escribir(f"Equipo A: {conteo['Equipo A']}")
        escribir(f"Equipo B: {conteo['Equipo B']}")
        escribir(
            f"Sin clasificar: "
            f"{conteo['Sin clasificar']}"
        )
        escribir(f"Descartadas: {conteo['Descartado']}")

        escribir(
            f"Imagenes guardadas: "
            f"{sum(bool(r['archivo']) and r['archivo'] != 'Error al guardar' for r in registros)}"
        )

        # ==================================
        # RESUMEN POR FRAME
        # ==================================

        for frame_num in range(
            1, frames_procesados + 1
        ):

            datos = [
                r for r in registros
                if r["frame"] == frame_num
            ]

            conteo_frame = Counter(
                r["equipo"]
                for r in datos
            )

            escribir()
            escribir(f"FRAME {frame_num}")
            escribir("-" * 65)

            escribir(f"Personas detectadas: {len(datos)}")
            escribir(
                f"Equipo A: {conteo_frame['Equipo A']}"
            )
            escribir(
                f"Equipo B: {conteo_frame['Equipo B']}"
            )
            escribir(
                f"Sin clasificar: "
                f"{conteo_frame['Sin clasificar']}"
            )
            escribir(
                f"Descartadas: "
                f"{conteo_frame['Descartado']}"
            )

            escribir()
            escribir("DETALLE DE DETECCIONES")

            for r in datos:

                escribir()
                escribir(
                    f"Persona {r['deteccion']:02d}"
                )

                escribir(f"Equipo: {r['equipo']}")
                escribir(
                    f"Confianza YOLO: "
                    f"{r['confianza'] * 100:.1f}%"
                )
                escribir(f"Bounding box: {r['box']}")
                escribir(
                    f"Ancho: {r['ancho']} px"
                )
                escribir(
                    f"Alto: {r['alto']} px"
                )
                escribir(
                    f"Area de pantalla: "
                    f"{r['area_pct']:.2f}%"
                )

                if r["brillo"] is not None:

                    escribir(
                        f"Luminosidad: "
                        f"{r['brillo'] * 100:.1f}%"
                    )

                    escribir(
                        f"Saturacion: "
                        f"{r['saturacion'] * 100:.1f}%"
                    )

                if r["margen"] is not None:

                    escribir(
                        f"Margen de agrupamiento: "
                        f"{r['margen']:.3f}"
                    )

                escribir(f"Motivo: {r['motivo']}")
                escribir(
                    f"Imagen: "
                    f"{r['archivo'] or 'No guardada'}"
                )

    print(f"TXT generado: {ruta.resolve()}")


# ==========================================
# GENERAR CSV
# ==========================================

def generar_csv(registros):

    ruta = CARPETA_SALIDA / "diagnostico.csv"

    columnas = [
        "frame",
        "deteccion",
        "equipo",
        "confianza",
        "box",
        "ancho",
        "alto",
        "area_pct",
        "brillo",
        "saturacion",
        "margen",
        "motivo",
        "archivo"
    ]

    with open(
        ruta,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as archivo:

        writer = csv.DictWriter(
            archivo,
            fieldnames=columnas
        )

        writer.writeheader()

        for r in registros:

            writer.writerow({
                columna: r[columna]
                for columna in columnas
            })


# ==========================================
# PROCESAR TRES FRAMES
# ==========================================

def procesar():

    CARPETA_SALIDA.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Cargando YOLOv8...")

    modelo = YOLO(MODELO_YOLO)

    cap = cv2.VideoCapture(VIDEO)

    if not cap.isOpened():

        print("No se pudo abrir el video.")
        return

    registros = []
    frames_procesados = 0

    for numero_frame in range(
        1, CANTIDAD_FRAMES + 1
    ):

        ret, frame = cap.read()

        if not ret:
            break

        frames_procesados += 1

        print(
            f"Procesando frame {numero_frame}..."
        )

        resultado = modelo.predict(
            frame,
            classes=[0],
            conf=CONFIANZA_YOLO,
            imgsz=YOLO_IMGSZ,
            verbose=False
        )[0]

        if resultado.boxes is None:
            continue

        boxes = resultado.boxes.xyxy.cpu().numpy()
        confianzas = resultado.boxes.conf.cpu().numpy()

        alto_frame, ancho_frame = frame.shape[:2]

        for numero_deteccion, (
            box, confianza
        ) in enumerate(
            zip(boxes, confianzas),
            start=1
        ):

            x1, y1, x2, y2 = map(int, box)

            ancho_box = max(0, x2 - x1)
            alto_box = max(0, y2 - y1)

            area_pct = (
                ancho_box * alto_box
                / (ancho_frame * alto_frame)
                * 100
            )

            aceptado = True
            motivo = "Deteccion valida"

            if FILTRAR_JUGADORES:

                aceptado, motivo = posible_jugador(
                    box,
                    frame
                )

            recorte = recortar_persona(
                frame,
                box
            )

            if recorte is None:

                aceptado = False
                motivo = "Bounding box invalido"

            caracteristicas = None
            brillo = None
            saturacion = None

            if aceptado:

                camiseta = extraer_camiseta(
                    recorte
                )

                datos = obtener_caracteristicas(
                    camiseta
                )

                if datos is None:

                    aceptado = False
                    motivo = "Camiseta invalida"

                else:

                    caracteristicas, brillo, saturacion = datos

            registros.append({
                "frame": numero_frame,
                "deteccion": numero_deteccion,
                "aceptado": aceptado,
                "equipo": (
                    "Sin clasificar"
                    if aceptado
                    else "Descartado"
                ),
                "confianza": float(confianza),
                "box": [x1, y1, x2, y2],
                "ancho": ancho_box,
                "alto": alto_box,
                "area_pct": area_pct,
                "brillo": brillo,
                "saturacion": saturacion,
                "caracteristicas": caracteristicas,
                "margen": None,
                "motivo": motivo,
                "imagen": recorte,
                "archivo": None
            })

    cap.release()

    # Clasificar los dos grupos de camisetas
    clasificar_equipos(registros)

    # Guardar recortes
    guardar_imagenes(registros)

    # Generar archivos de diagnostico
    generar_txt(
        registros,
        frames_procesados
    )

    generar_csv(registros)

    resumen = Counter(
        r["equipo"]
        for r in registros
    )

    print("\nPROCESAMIENTO FINALIZADO")
    print(f"Frames: {frames_procesados}")
    print(f"Detecciones: {len(registros)}")
    print(f"Equipo A: {resumen['Equipo A']}")
    print(f"Equipo B: {resumen['Equipo B']}")
    print(
        f"Sin clasificar: "
        f"{resumen['Sin clasificar']}"
    )
    print(
        f"Descartadas: "
        f"{resumen['Descartado']}"
    )
    print(
        f"Carpeta: "
        f"{CARPETA_SALIDA.resolve()}"
    )


# ==========================================
# INICIAR
# ==========================================

if __name__ == "__main__":
    procesar()
