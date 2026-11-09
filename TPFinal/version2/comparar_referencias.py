"""Comparar IDs nuevos con las muestras (equipo,dorsal) verificadas a mano.

Evalua solo esas muestras. Menos IDs dibujados no se confunde con mayor
exactitud: informa faltantes, duplicados y equipos incorrectos por separado.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from seguimiento import SeguimientoJugadores


def cargar(ruta):
    with open(ruta,encoding='utf-8') as f:
        return list(csv.DictReader(f))


def comparar(ruta):
    referencias=cargar('auditoria_ids/referencias.csv')
    original={(int(r['frame']),int(r['id'])):r
              for r in cargar('tracking_mejorado/trayectorias.csv')}
    nuevos=defaultdict(list)
    for r in cargar(ruta):
        nuevos[round(float(r['segundo_video'])*1000)].append(r)
    limite = (min(nuevos), max(nuevos)) if nuevos else None
    filas=[]
    for ref in referencias:
        origen=original[(int(ref['frame']),int(ref['id_observado']))]
        box=np.array([float(origen[k]) for k in ('x1','y1','x2','y2')])
        candidatos=[]
        for r in nuevos[round(float(origen['segundo_video'])*1000)]:
            actual=np.array([float(r[k]) for k in ('x1','y1','x2','y2')])
            iou=SeguimientoJugadores.iou(box,actual)
            if iou>=.5:
                candidatos.append((r,iou))
        equipo=0 if ref['equipo']=='A' else 1
        compatibles=[(r,i) for r,i in candidatos if int(r['equipo_bloqueado'])==equipo]
        estado=('asignado' if len(compatibles)==1 else 'duplicado_o_ambiguo' if len(compatibles)>1
                else 'equipo_incorrecto' if candidatos else 'sin_id')
        instante = round(float(origen['segundo_video'])*1000)
        if limite and not limite[0] <= instante <= limite[1]:
            estado = 'fuera_tramo'
        filas.append(dict(equipo=ref['equipo'],dorsal=ref['dorsal'],segundo_video=origen['segundo_video'],
                          id_anterior=ref['id_observado'],
                          id_nuevo=compatibles[0][0]['id'] if estado=='asignado' else '',
                          estado=estado,ids_coincidentes=';'.join(r['id'] for r,_ in candidatos)))
    destino=Path('auditoria_ids')/f'comparacion_{Path(ruta).parent.name}.csv'
    with open(destino,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(filas[0]));w.writeheader();w.writerows(filas)
    for identidad in sorted({(r['equipo'],r['dorsal']) for r in filas}):
        muestras=[r for r in filas if (r['equipo'],r['dorsal'])==identidad]
        ids={r['id_nuevo'] for r in muestras if r['estado']=='asignado'}
        asignadas=sum(r['estado']=='asignado' for r in muestras)
        print(identidad, 'IDs:',sorted(ids),'asignadas:',asignadas,'/',len(muestras),
              'intercambio:',len(ids)>1,'observaciones:',[r['estado'] for r in muestras])
    print('Detalle:',destino)
    return filas


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--trayectorias',default='tracking_mejorado/trayectorias.csv')
    comparar(parser.parse_args().trayectorias)
