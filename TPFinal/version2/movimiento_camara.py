"""Compensacion economica de paneo y zoom con puntos del fondo."""
import cv2
import numpy as np


class MovimientoCamara:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=400, edgeThreshold=12)
        self.anterior = None

    def actualizar(self, frame):
        alto, ancho = frame.shape[:2]
        mini = cv2.resize(frame, (480, max(1, round(alto * 480 / ancho))))
        gris = cv2.cvtColor(mini, cv2.COLOR_BGR2GRAY)
        mascara = np.zeros_like(gris)
        # Gradas y publicidad: menos jugadores moviendose independientemente.
        mascara[:int(gris.shape[0] * .48)] = 255
        puntos, descriptores = self.orb.detectAndCompute(gris, mascara)
        anterior = self.anterior
        self.anterior = (puntos, descriptores)
        if anterior is None or anterior[1] is None or descriptores is None:
            return None
        vecinos = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(anterior[1], descriptores, k=2)
        buenos = [par[0] for par in vecinos if len(par) == 2 and par[0].distance < .75 * par[1].distance]
        if len(buenos) < 12:
            return None
        origen = np.float32([anterior[0][m.queryIdx].pt for m in buenos])
        destino = np.float32([puntos[m.trainIdx].pt for m in buenos])
        matriz, inliers = cv2.estimateAffinePartial2D(origen, destino, method=cv2.RANSAC,
                                                     ransacReprojThreshold=2.0)
        if matriz is None or int(inliers.sum()) < 10 or float(inliers.mean()) < .40:
            return None
        escala = float(np.hypot(matriz[0, 0], matriz[1, 0]))
        if not .85 <= escala <= 1.18 or np.linalg.norm(matriz[:, 2]) > 120:
            return None
        matriz[:, 2] *= ancho / 480
        return matriz
