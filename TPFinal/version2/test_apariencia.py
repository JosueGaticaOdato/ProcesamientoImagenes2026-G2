import unittest

import cv2
import numpy as np

from apariencia import obtener_contorno_cancha, clasificar_uniforme_azul_blanco, pies_sobre_piso
from movimiento_camara import MovimientoCamara
from PIL import Image


class CanchaTests(unittest.TestCase):
    def test_camiseta_blanca_dominante_con_azul_por_solapamiento(self):
        rgb = np.full((40, 30, 3), 200, dtype=np.uint8)
        rgb[:14] = (20, 30, 100)
        self.assertEqual(clasificar_uniforme_azul_blanco(Image.fromarray(rgb)), 1)
    def test_camiseta_clara_y_pantalon_negro_no_es_equipo_blanco(self):
        camiseta = Image.fromarray(np.full((40, 30, 3), 200, dtype=np.uint8))
        shorts = np.full((30, 30, 3), 25, dtype=np.uint8)
        self.assertIsNone(clasificar_uniforme_azul_blanco(camiseta, shorts=shorts))

    def test_camiseta_azul_y_short_blanco_queda_ambiguo(self):
        camiseta = Image.fromarray(np.full((40, 30, 3), (20, 30, 100), dtype=np.uint8))
        shorts = np.full((30, 30, 3), 200, dtype=np.uint8)
        self.assertIsNone(clasificar_uniforme_azul_blanco(camiseta, shorts=shorts))
    def test_pies_sobre_pintura_y_madera_pero_no_banda_azul(self):
        for bgr, esperado in [((150, 180, 210), True), ((0, 0, 180), True),
                              ((100, 25, 15), False)]:
            frame = np.full((100, 100, 3), bgr, dtype=np.uint8)
            self.assertEqual(pies_sobre_piso(frame, [30, 10, 60, 70]), esperado)

    def test_estima_paneo_zoom_sobre_fondo(self):
        rng = np.random.default_rng(42)
        fondo = rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)
        matriz = np.array([[1.02, 0., 12.], [0., 1.02, 5.]])
        siguiente = cv2.warpAffine(fondo, matriz, (640, 360))
        camara = MovimientoCamara()
        self.assertIsNone(camara.actualizar(fondo))
        estimada = camara.actualizar(siguiente)
        self.assertIsNotNone(estimada)
        np.testing.assert_allclose(estimada[:, 2], matriz[:, 2], atol=2.0)
    def test_uniformes_calibrados_y_negro_ambiguo(self):
        for rgb, esperado in [((20, 30, 100), 0), ((200, 200, 200), 1),
                              ((90, 90, 90), 2), ((15, 15, 15), None)]:
            camiseta = Image.fromarray(np.full((40, 30, 3), rgb, dtype=np.uint8))
            self.assertEqual(clasificar_uniforme_azul_blanco(camiseta), esperado)
    def test_sin_parquet_no_impone_filtro(self):
        self.assertIsNone(obtener_contorno_cancha(np.zeros((180, 320, 3), np.uint8)))

    def test_incluye_zona_pintada_pero_excluye_grada(self):
        frame = np.zeros((180, 320, 3), np.uint8)
        frame[65:155, 15:305] = (150, 180, 210)
        frame[95:130, 100:180] = (0, 0, 180)
        cancha = obtener_contorno_cancha(frame)
        self.assertIsNotNone(cancha)
        self.assertGreaterEqual(cv2.pointPolygonTest(cancha, (140, 110), False), 0)
        self.assertLess(cv2.pointPolygonTest(cancha, (140, 175), False), 0)


if __name__ == '__main__':
    unittest.main()
