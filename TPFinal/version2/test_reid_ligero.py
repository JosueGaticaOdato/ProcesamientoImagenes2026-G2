import unittest

import numpy as np
import torch

from reid_ligero import Reidentificador, RUTA_PESOS


@unittest.skipUnless(RUTA_PESOS.is_file(), 'Pesos OSNet no disponibles')
class ReidentificadorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.extractor = Reidentificador('cpu')

    def test_pesos_reales_vectores_normalizados_y_lote_independiente(self):
        rng = np.random.default_rng(4)
        a = rng.integers(0, 256, (180, 60, 3), dtype=np.uint8)
        b = rng.integers(0, 256, (220, 80, 3), dtype=np.uint8)
        juntos = self.extractor.extraer([a, b])
        solo = self.extractor.extraer([a])
        self.assertEqual(juntos.shape, (2, 512))
        self.assertTrue(np.isfinite(juntos).all())
        np.testing.assert_allclose(np.linalg.norm(juntos, axis=1), 1, atol=1e-5)
        np.testing.assert_allclose(juntos[0], solo[0], atol=1e-5)
