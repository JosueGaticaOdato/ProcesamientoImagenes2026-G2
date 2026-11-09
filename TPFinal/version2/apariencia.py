"""Recortes de apariencia y lectura conservadora del dorsal a resolucion nativa."""
import warnings

import cv2
import numpy as np
from PIL import Image


def obtener_contorno_cancha(frame):
    """Aproximar parquet en tomas abiertas, sin aplicar una ROI fija."""
    alto, ancho = frame.shape[:2]
    mini = cv2.resize(frame, (320, max(1, round(alto * 320 / ancho))))
    hsv = cv2.cvtColor(mini, cv2.COLOR_BGR2HSV)
    mascara = cv2.inRange(hsv, np.array([8, 20, 75]), np.array([40, 180, 255]))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None
    mayor = max(contornos, key=cv2.contourArea)
    if cv2.contourArea(mayor) < mini.shape[0] * mini.shape[1] * .15:
        return None
    return cv2.convexHull(mayor).astype(np.float32) * (ancho / 320)


def clasificar_uniforme_azul_blanco(camiseta, devolver_confianza=False, shorts=None):
    """Reglas de color calibradas para azul, blanco y gris en este video."""
    hsv = cv2.cvtColor(np.asarray(camiseta), cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    azul = float(np.mean((h >= 90) & (h <= 140) & (s > 60) & (v > 30)))
    blanco = float(np.mean((s < 60) & (v > 140)))
    gris = float(np.mean((s < 60) & (v >= 45) & (v <= 140)))
    brillo = float(np.median(v))
    equipo, fuerte = None, False
    if blanco > .30 and gris < .22 and brillo > 120:
        equipo, fuerte = 1, True
    elif azul > .08:
        equipo, fuerte = 0, True
    elif gris > .24 and blanco < .20 and brillo > 45:
        equipo, fuerte = 2, True
    elif 30 < brillo < 120 and blanco < .25 and gris < .22:
        equipo = 0
    if shorts is not None and equipo in (0, 1):
        _, ss, vv = cv2.split(cv2.cvtColor(shorts, cv2.COLOR_BGR2HSV))
        brillo_short = float(np.median(vv))
        blanco_short = float(np.mean((ss < 70) & (vv > 140)))
        # En este partido ambos equipos llevan camiseta y shorts del mismo tono.
        # Brazos claros sobre remera negra no deben convertir a un entrenador en B.
        if (equipo == 1 and brillo_short < 90) or (
                equipo == 0 and brillo_short > 120 and blanco_short > .35):
            equipo, fuerte = None, False
    return (equipo, fuerte) if devolver_confianza else equipo


def pies_sobre_piso(frame, box):
    """Comprobar debajo de los pies: parquet o pintura roja de esta cancha."""
    x1, y1, x2, y2 = map(int, box)
    alto, ancho = frame.shape[:2]
    centro = (x1 + x2) // 2
    radio = max(6, min(20, (x2 - x1) // 3))
    ya, yb = max(0, y2 + 2), min(alto, y2 + max(8, (y2 - y1) // 12))
    if ya >= yb:
        return False
    parche = frame[ya:yb, max(0, centro-radio):min(ancho, centro+radio)]
    if not parche.size:
        return False
    h, s, v = cv2.split(cv2.cvtColor(parche, cv2.COLOR_BGR2HSV))
    madera = (h >= 8) & (h <= 40) & (s >= 20) & (s <= 180) & (v >= 75)
    pintura = ((h <= 10) | (h >= 170)) & (s >= 90) & (v >= 70)
    return float(np.mean(madera | pintura)) >= .20


def recortes_persona(frame, box):
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    persona = frame[y1:y2, x1:x2]
    if persona.size == 0:
        return {}
    alto, ancho = persona.shape[:2]
    zonas = {'cuerpo': (0, 1, 0, 1), 'cabeza': (.12, .88, 0, .22),
             'shorts': (.12, .88, .48, .76), 'dorsal': (.15, .85, .16, .63)}
    salida = {}
    for nombre, (xa, xb, ya, yb) in zonas.items():
        recorte = persona[int(ya * alto):int(yb * alto), int(xa * ancho):int(xb * ancho)]
        if recorte.size and min(recorte.shape[:2]) >= (24 if nombre == 'cabeza' else 16):
            if nombre == 'shorts' or cv2.Laplacian(cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() >= 15:
                salida[nombre] = recorte.copy()
    return salida


def imagen_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def descriptor_rapido(bgr):
    """Color y textura por franjas, sin ejecutar una red por cada parte."""
    bloques = []
    for franja in np.array_split(bgr, 3, axis=0):
        hsv = cv2.cvtColor(franja, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).ravel()
        brillo = cv2.calcHist([hsv], [2], None, [8], [0, 256]).ravel()
        gris = cv2.resize(cv2.cvtColor(franja, cv2.COLOR_BGR2GRAY), (8, 8))
        textura = np.abs(cv2.Laplacian(gris, cv2.CV_32F)).ravel()
        for vector in (hist, brillo, textura):
            bloques.append(vector / max(float(np.linalg.norm(vector)), 1e-8))
    vector = np.concatenate(bloques).astype(np.float32)
    return vector / max(float(np.linalg.norm(vector)), 1e-8)


class LectorDorsal:
    def __init__(self, gpu=False, activo=True):
        self.reader = None
        if activo:
            try:
                import easyocr
                self.reader = easyocr.Reader(['en'], gpu=gpu)
            except (ImportError, OSError, RuntimeError) as error:
                warnings.warn(f'OCR desactivado: {error}. Instalar easyocr para leer dorsales.')

    @staticmethod
    def filtrar_lecturas(resultados, altura_imagen):
        candidatos = []
        for poligono, texto, confianza in resultados:
            puntos = np.asarray(poligono)
            proporcion_alto = np.ptp(puntos[:, 1]) / altura_imagen
            if texto.isdigit() and 1 <= len(texto) <= 2 and confianza >= .85 and proporcion_alto >= .20:
                candidatos.append(texto)
        # Varios numeros en la region no constituyen una lectura fiable.
        return candidatos[0] if len(candidatos) == 1 else None

    def leer(self, torso):
        if self.reader is None or torso is None or torso.shape[1] < 18 or torso.shape[0] < 24:
            return None
        ampliado = cv2.resize(torso, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        resultados = self.reader.readtext(ampliado, allowlist='0123456789', detail=1,
                                          paragraph=False, canvas_size=512)
        return self.filtrar_lecturas(resultados, ampliado.shape[0])
