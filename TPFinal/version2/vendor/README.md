OSNet copiado de https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/models/osnet.py
Licencia MIT: `LICENSE.osnet.txt`, copyright Kaiyang Zhou.

SHA256 del codigo obtenido: c7c1c29187d6330f859c91da229271531920464c7011aec13842a086b2263cae.
No se invoca la descarga interna ni el cargador de pesos del archivo original.
El adaptador `reid_ligero.py` carga unicamente tensores con `weights_only=True`.

Pesos oficiales del autor (MSMT17, no ImageNet):
https://huggingface.co/kaiyangzhou/osnet/blob/main/osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth

Archivo local: `modelos/osnet_x025_msmt17.pth` (9,336,983 bytes).
SHA256: cf55163d78fc44c62c82f85ab62d39f10438679b5abe8c698ae08cfa84aa6e18.
En ejecucion no se descarga ningun archivo ni se requiere instalar torchreid.
