"""Extractor OSNet x0.25; pesos oficiales MSMT17, sin descargas en ejecucion."""
from pathlib import Path

import cv2
import numpy as np
import torch


RUTA_PESOS = Path(__file__).with_name('modelos') / 'osnet_x025_msmt17.pth'


class Reidentificador:
    def __init__(self, device='cpu'):
        from vendor.osnet import osnet_x0_25
        self.device = device
        self.modelo = osnet_x0_25(num_classes=1, pretrained=False)
        pesos = torch.load(RUTA_PESOS, map_location='cpu', weights_only=True)
        pesos = pesos.get('state_dict', pesos)
        pesos = {k.removeprefix('module.'): v for k, v in pesos.items()}
        pesos = {k: v for k, v in pesos.items() if not k.startswith('classifier.')}
        faltantes, extras = self.modelo.load_state_dict(pesos, strict=False)
        if extras or set(faltantes) != {'classifier.weight', 'classifier.bias'}:
            raise RuntimeError(f'Pesos OSNet incompatibles: {faltantes}, {extras}')
        self.modelo = self.modelo.to(device).eval()
        self.media = torch.tensor([.485, .456, .406], device=device).view(1, 3, 1, 1)
        self.desviacion = torch.tensor([.229, .224, .225], device=device).view(1, 3, 1, 1)

    def extraer(self, recortes):
        if not recortes:
            return np.empty((0, 512), dtype=np.float32)
        imagenes = [cv2.cvtColor(cv2.resize(r, (128, 256)), cv2.COLOR_BGR2RGB)
                    for r in recortes]
        lote = torch.from_numpy(np.stack(imagenes)).permute(0, 3, 1, 2).float().to(self.device) / 255
        lote = (lote - self.media) / self.desviacion
        with torch.inference_mode():
            vectores = torch.nn.functional.normalize(self.modelo(lote), dim=1)
        return vectores.cpu().numpy().astype(np.float32)
