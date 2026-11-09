"""Identidades persistentes con correlacion temporal y cupos por equipo."""
from collections import Counter, deque

import numpy as np
from scipy.optimize import linear_sum_assignment


class SeguimientoJugadores:
    def __init__(self, ventana=10, memoria=90):
        self.ventana = ventana
        self.memoria = memoria
        self.tracks = []
        self.recuperaciones = 0
        self.propietarios_raw = {}

    @staticmethod
    def centro(box):
        return (box[:2] + box[2:]) / 2

    @staticmethod
    def iou(a, b):
        lados = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
        inter = float(np.prod(lados))
        union = float(np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - inter)
        return inter / max(union, 1e-8)

    @classmethod
    def indices_sin_duplicados(cls, cajas, confianzas, equipos=None, umbral=.65):
        """Una misma persona no debe consumir dos IDs por recuadros duplicados.

        No fusionar por geometria si se observaron equipos distintos o desconocidos.
        """
        elegidos = []
        for i in sorted(range(len(cajas)), key=lambda k: -float(confianzas[k])):
            duplicado = any(
                (equipos is None or (equipos[i] is not None and equipos[i] == equipos[j]))
                and cls.iou(cajas[i], cajas[j]) > umbral for j in elegidos)
            if not duplicado:
                elegidos.append(i)
        return sorted(elegidos)

    def raw_coincide(self, track, raw_id, frame):
        if raw_id < 0:
            return False
        propietario = self.propietarios_raw.get(raw_id)
        if track['id'] is not None and propietario is not None and propietario != track['id']:
            return False
        ultimo = track['historia'][-1]
        return (ultimo['raw_id'] == raw_id or
                frame - track.get('aliases', {}).get(raw_id, -1000000) <= self.memoria)

    def conocido(self, raw_id, box, frame):
        for track in self.tracks:
            if track['id'] is None or not track.get('espacio_valido', True):
                continue
            ultimo = track['historia'][-1]
            dt = frame - ultimo['frame']
            if dt > 15:
                continue
            alto = max(1., ultimo['box'][3] - ultimo['box'][1])
            proporcion = (box[3] - box[1]) / alto
            distancia = np.linalg.norm(self.centro(box) - self.centro(ultimo['box'])) / alto
            raw_familiar = self.raw_coincide(track, raw_id, frame)
            geometria_cercana = (dt <= 3 and distancia <= .6
                                and self.iou(ultimo['box'], box) >= .35)
            if (raw_familiar or geometria_cercana) and .5 <= proporcion <= 2.0 and distancia <= 2.0:
                return True
        return False

    def cambio_de_plano(self):
        self.tracks = [t for t in self.tracks if t['id'] is not None]
        self.propietarios_raw.clear()
        for track in self.tracks:
            track['espacio_valido'] = False
            track['aliases'] = {}

    def predecir(self, frame, ancho, alto, visibles, max_frames=5):
        ids_visibles = {d['jugador_id'] for d in visibles}
        predicciones = []
        for track in self.tracks:
            if track['id'] is None or track['id'] in ids_visibles or not track.get('espacio_valido', True):
                continue
            historia = list(track['historia'])
            ultimo = historia[-1]
            gap = frame - ultimo['frame']
            if not 1 <= gap <= max_frames:
                continue
            velocidad = np.zeros(2)
            if len(historia) >= 2:
                primero = historia[max(0, len(historia) - 3)]
                velocidad = (self.centro(ultimo['box']) - self.centro(primero['box'])) / max(
                    1, ultimo['frame'] - primero['frame'])
            box = ultimo['box'].copy() + np.tile(velocidad * gap, 2)
            box[[0, 2]] = np.clip(box[[0, 2]], 0, ancho - 1)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, alto - 1)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            if any(self.iou(box, d['box']) > .5 for d in visibles):
                continue
            predicciones.append(dict(ultimo, box=box, jugador_id=track['id'],
                                     equipo=track['equipo'], dorsal=self.dorsal_confirmado(track),
                                     frames_predichos=gap))
        return predicciones

    def compensar_camara(self, matriz):
        if matriz is None:
            return
        vistos = set()
        for track in self.tracks:
            for muestra in track['historia']:
                if id(muestra) in vistos:
                    continue
                vistos.add(id(muestra))
                x1, y1, x2, y2 = muestra['box']
                esquinas = np.array([[x1, y1, 1], [x2, y1, 1],
                                     [x2, y2, 1], [x1, y2, 1]]) @ matriz.T
                muestra['box'] = np.r_[esquinas.min(axis=0), esquinas.max(axis=0)]

    @staticmethod
    def dorsal_confirmado(track):
        if track.get('dorsal_fijo') is not None:
            return track['dorsal_fijo']
        votos = Counter(track.get('dorsales', []))
        if not votos:
            return None
        numero, cantidad = votos.most_common(1)[0]
        return numero if cantidad >= 3 and cantidad / sum(votos.values()) >= .8 else None

    def propietario_dorsal(self, equipo, numero):
        if equipo is None or numero is None:
            return None
        coincidencias = [t for t in self.tracks if t['id'] is not None and
                         t['equipo'] == equipo and self.dorsal_confirmado(t) == numero]
        return coincidencias[0] if len(coincidencias) == 1 else None

    @staticmethod
    def similitud_reid(track, deteccion):
        actual = deteccion.get('reid')
        if actual is None:
            return None
        referencias = list(track.get('anclas', [])) + list(track.get('vistas_dorsal', []))
        if not referencias:
            historia = list(track.get('galeria', [])) or list(track['historia'])
            referencias = [m['reid'] for m in historia if m.get('reid') is not None]
        if not referencias:
            return None
        return float(max(v @ actual for v in referencias))

    def memoria_confiable(self, track, deteccion):
        if (deteccion.get('dorsal_verificado', False) and deteccion.get('dorsal') is not None
                and deteccion['dorsal'] == self.dorsal_confirmado(track)
                and deteccion.get('equipo') == track['equipo']):
            return True
        propio = self.similitud_reid(track, deteccion)
        if propio is None:
            return True
        rivales = [self.similitud_reid(t, deteccion) for t in self.tracks
                   if t is not track and t['id'] is not None and t['equipo'] == track['equipo']
                   and t.get('anclas')]
        rivales = [s for s in rivales if s is not None]
        return propio >= .75 and (not rivales or propio - max(rivales) >= .04)

    def fijar_ancla(self, track):
        if track.get('anclas'):
            return
        # La primera vista limpia queda fija: una asociacion posterior no la reemplaza.
        for muestra in track['galeria']:
            if muestra.get('reid') is not None and not muestra.get('ocluido', False):
                track['anclas'] = [muestra['reid'].copy()]
                return

    def costo_reidentificacion(self, reservado, candidato, equipo):
        """Comparar trayectoria nueva estable con galeria de identidad perdida."""
        if reservado['equipo'] != equipo:
            return 1000.0
        a = list(reservado.get('galeria', [])) or list(reservado['historia'])
        b = list(candidato['historia'])
        dorsal_a = self.dorsal_confirmado(reservado)
        dorsal_b = self.dorsal_confirmado(candidato)
        if dorsal_a is not None and dorsal_b is not None and dorsal_a != dorsal_b:
            return 1000.0
        mismo_dorsal = dorsal_a is not None and dorsal_a == dorsal_b
        actuales_reid = [m['reid'] for m in b if m.get('reid') is not None]
        similitud_profunda = None
        if actuales_reid:
            actual = np.mean(actuales_reid, axis=0)
            actual /= max(float(np.linalg.norm(actual)), 1e-8)
            similitud_profunda = self.similitud_reid(reservado, {'reid': actual})
        # Media de las observaciones nuevas y mejor vista historica por region.
        similitudes = []
        for region in ('cuerpo', 'shorts', 'cabeza'):
            anteriores = [m.get('partes', {}).get(region) for m in a]
            actuales = [m.get('partes', {}).get(region) for m in b]
            anteriores = [v for v in anteriores if v is not None]
            actuales = [v for v in actuales if v is not None]
            if anteriores and actuales:
                actual = np.mean(actuales, axis=0)
                actual /= max(float(np.linalg.norm(actual)), 1e-8)
                valores = sorted(float(v @ actual) for v in anteriores)
                similitudes.append(float(np.mean(valores[-3:])))
        if len(similitudes) < 2 and not mismo_dorsal:
            return 1000.0
        regional = float(np.mean(similitudes)) if similitudes else 0.0
        if similitud_profunda is not None:
            if similitud_profunda < .65 or (regional < .65 and not mismo_dorsal):
                return 1000.0
            return .85 * (1 - similitud_profunda) + .15 * (1 - regional) - (.2 if mismo_dorsal else 0)
        vector = np.mean([m['vector'] for m in b], axis=0)
        vector /= max(float(np.linalg.norm(vector)), 1e-8)
        apariencia = max(float(m['vector'] @ vector) for m in a)
        if not mismo_dorsal and (regional < .88 or apariencia < .80):
            return 1000.0
        return .7 * (1 - regional) + .3 * (1 - apariencia) - (.2 if mismo_dorsal else 0)

    def recuperar(self, candidato, equipo, frame):
        ultima = candidato['historia'][-1]
        numero_actual = ultima.get('dorsal') if ultima.get('dorsal_verificado',False) and not ultima.get('ocluido',False) else None
        propietario = self.propietario_dorsal(equipo, numero_actual)
        if propietario is not None:
            if propietario.get('asignado_frame') != frame and frame-propietario['historia'][-1]['frame'] > 2:
                return propietario
            return None
        perdidos = [t for t in self.tracks if t['id'] is not None and
                    t.get('asignado_frame') != frame and
                    frame - t['historia'][-1]['frame'] > 2]
        puntuados = sorted([(self.costo_reidentificacion(t, candidato, equipo), t)
                            for t in perdidos], key=lambda par: par[0])
        usa_reid = any(m.get('reid') is not None for m in candidato['historia'])
        limite = .32 if usa_reid else .15
        if not puntuados or puntuados[0][0] >= limite:
            return None
        # No adivinar entre dos companeros con apariencia indistinguible.
        if len(puntuados) > 1 and puntuados[1][0] - puntuados[0][0] < .025:
            return None
        return puntuados[0][1]

    def costo(self, track, deteccion, frame):
        numero = deteccion.get('dorsal')
        if deteccion.get('dorsal_verificado', False) and numero is not None and not deteccion.get('ocluido',False):
            propietario = self.propietario_dorsal(deteccion['equipo'], numero)
            if track['id'] is not None and propietario is not None and propietario is not track:
                return 1000.0
            referencia = self.dorsal_confirmado(track)
            if referencia is not None and referencia != numero:
                return 1000.0
            if referencia == numero and track['equipo'] is not None and track['equipo'] == deteccion['equipo']:
                similitud = self.similitud_reid(track, deteccion)
                if similitud is not None and similitud < .40:
                    return 1000.0
                distancia = np.linalg.norm(self.centro(track['historia'][-1]['box']) - self.centro(deteccion['box']))
                return .02 + .01 * min(2., distancia / max(40., deteccion['box'][3]-deteccion['box'][1]))
        if not track.get('espacio_valido', True):
            return 1000.0
        historia = list(track['historia'])
        ultimo = historia[-1]
        dt = frame - ultimo['frame']
        mismo_raw = self.raw_coincide(track, deteccion['raw_id'], frame)
        dorsal = self.dorsal_confirmado(track)
        mismo_dorsal = dorsal is not None and dorsal == deteccion.get('dorsal')
        # Un ID reservado no debe quedar irrecuperable tras tres segundos.
        recuperacion = dt > self.memoria
        if recuperacion and not (mismo_dorsal or mismo_raw):
            return 1000.0
        velocidad = np.zeros(2)
        if len(historia) > 1:
            # Ventana corta para responder a aceleraciones y cambios de direccion.
            primero = historia[max(0, len(historia) - 3)]
            velocidad = (self.centro(ultimo['box']) - self.centro(primero['box'])) / max(
                1, ultimo['frame'] - primero['frame'])
        prediccion = self.centro(ultimo['box']) + velocidad * min(dt, self.ventana)
        escala = max(40.0, ultimo['box'][3] - ultimo['box'][1])
        distancia = np.linalg.norm(self.centro(deteccion['box']) - prediccion) / escala
        apariencia = np.mean([
            float(np.dot(m['vector'], deteccion['vector'])) for m in historia
        ])
        color = np.mean([
            np.minimum(m['histograma'], deteccion['histograma']).sum() for m in historia
        ])
        reid = self.similitud_reid(track, deteccion)
        if reid is not None and reid < .40:
            return 1000.0
        costo_reid = .35 * (1 - reid) if reid is not None else 0.0
        contrarios_visibles = sum(m.get('equipo') is not None and m['equipo'] != track['equipo']
                                 and not m.get('ocluido',False) for m in historia[-8:])
        if track['id'] is not None and reid is not None and reid < .75 and color < .5 and contrarios_visibles >= 5:
            return 1000.0
        if reid is not None and track['id'] is not None:
            rivales = [self.similitud_reid(t, deteccion) for t in self.tracks
                       if t is not track and t['id'] is not None
                       and t.get('anclas')]
            rivales = [s for s in rivales if s is not None]
            if rivales and max(rivales) >= .80 and max(rivales) - reid >= .08:
                return 1000.0
        proporcion = (deteccion['box'][3] - deteccion['box'][1]) / escala
        opuesto = (track['equipo'] is not None and deteccion['equipo'] is not None
                   and track['equipo'] != deteccion['equipo']
                   and deteccion.get('equipo_fuerte', False))
        # Continuidad corta: no exigir apariencia nueva ni diez votos otra vez.
        continuo = mismo_raw and dt <= 15 and distancia <= 1.5 and .5 <= proporcion <= 2.0
        if continuo:
            if opuesto and (color < .35 or track.get('contrarios', 0) >= 2):
                return 1000.0
            regiones = [1 - float(a @ deteccion['partes'][nombre])
                        for nombre, a in (ultimo.get('partes') or {}).items()
                        if nombre in deteccion.get('partes', {})]
            regional_rapido = float(np.mean(regiones)) if regiones else 0.0
            return (.25 * min(distancia, 1.0) + .1 * (1 - color)
                    + .15 * regional_rapido + costo_reid + (.15 if opuesto else 0))
        if track['id'] is not None and deteccion.get('uniforme_contradictorio', False):
            return 1000.0
        # El ID crudo de ByteTrack solo aporta una pista, nunca define la identidad.
        bono = 0.18 if mismo_raw else 0.0
        limite_movimiento = min(3.0, 1.4 + 0.20 * dt) if mismo_raw else min(2.5, 1.0 + 0.15 * dt)
        solapamiento = self.iou(ultimo['box'], deteccion['box'])
        apariencia_minima = .35 if dt <= 3 and solapamiento >= .35 else .50
        if reid is not None and reid >= .75:
            apariencia_minima = .25
        if (not recuperacion and distancia > limite_movimiento) or apariencia < apariencia_minima:
            return 1000.0
        penalizacion_equipo = 0.0
        if track['equipo'] is not None and deteccion['equipo'] is not None:
            if track['equipo'] != deteccion['equipo']:
                if deteccion.get('equipo_fuerte', False):
                    return 1000.0
                # Un voto ruidoso no rompe una identidad con continuidad espacial.
                penalizacion_equipo = .35
                if recuperacion or not mismo_raw:
                    return 1000.0
        # Comparar cada region solo cuando ambos frames tienen un recorte util.
        distancias_regiones = []
        for region, peso in [('cuerpo', .45), ('shorts', .35), ('cabeza', .20)]:
            actual = deteccion.get('partes', {}).get(region)
            anteriores = [m.get('partes', {}).get(region) for m in historia]
            anteriores = [v for v in anteriores if v is not None]
            if actual is not None and anteriores:
                similitud = np.mean([float(np.dot(v, actual)) for v in anteriores])
                distancias_regiones.append((peso, 1 - np.clip(similitud, -1, 1)))
        regional = (sum(p * d for p, d in distancias_regiones) /
                    sum(p for p, _ in distancias_regiones)) if distancias_regiones else 1 - apariencia
        if recuperacion:
            if not distancias_regiones or regional > .12 or apariencia < .85:
                return 1000.0
            distancia = 0.5
        dorsal = self.dorsal_confirmado(track)
        lectura = deteccion.get('dorsal')
        # Una lectura individual ayuda, pero nunca veta una asociacion por si sola.
        penalizacion_dorsal = 0.0
        if dorsal is not None and lectura is not None:
            penalizacion_dorsal = -.15 if dorsal == lectura else .25
        # Durante continuidad de ByteTrack amortiguar errores de extrapolacion.
        distancia_costo = min(distancia, 1.0) if mismo_raw and dt <= 3 else distancia
        return (.40 * distancia_costo + .15 * (1 - apariencia) + .10 * (1 - color)
                + .35 * regional + costo_reid + penalizacion_dorsal + penalizacion_equipo - bono)

    def actualizar(self, detecciones, frame):
        # Los IDs confirmados quedan reservados incluso cuando desaparecen.
        self.tracks = [t for t in self.tracks if t['id'] is not None or
                       frame - t['historia'][-1]['frame'] < self.ventana]
        pares = []
        # Primero recuperar confirmados; los candidatos no les roban detecciones.
        disponibles = list(range(len(detecciones)))
        for confirmados in (True, False):
            indices = [i for i, t in enumerate(self.tracks)
                       if (t['id'] is not None) == confirmados]
            if not indices or not disponibles:
                continue
            costos = np.array([[self.costo(self.tracks[i], detecciones[j], frame)
                                for j in disponibles] for i in indices])
            # Columnas ficticias permiten no asignar sin distorsionar otros pares.
            matriz = np.concatenate([costos, np.full((len(indices), len(indices)), .85)], axis=1)
            filas, columnas = linear_sum_assignment(matriz)
            nuevos = [(indices[int(i)], disponibles[int(j)]) for i, j in zip(filas, columnas)
                      if j < len(disponibles) and costos[i, j] < .85]
            pares.extend(nuevos)
            usados = {j for _, j in nuevos}
            disponibles = [j for j in disponibles if j not in usados]
        asignados = {j for _, j in pares}
        for j in range(len(detecciones)):
            if j not in asignados:
                self.tracks.append({'id': None, 'equipo': None,
                                    'historia': deque(maxlen=self.ventana),
                                    'galeria': deque(maxlen=40),
                                    'dorsales': deque(maxlen=30), 'aliases': {},
                                    'espacio_valido': True, 'contrarios': 0, 'anclas': []})
                pares.append((len(self.tracks) - 1, j))
        visibles = []
        for i, _ in pares:
            self.tracks[i]['asignado_frame'] = frame
        for i, j in pares:
            track, deteccion = self.tracks[i], detecciones[j]
            anterior = track['historia'][-1] if track['historia'] else None
            if not track.get('espacio_valido', True):
                track['historia'].clear()
                track['espacio_valido'] = True
            muestra = dict(deteccion, frame=frame)
            opuesto = (track['equipo'] is not None and deteccion['equipo'] is not None
                       and track['equipo'] != deteccion['equipo']
                       and deteccion.get('equipo_fuerte', False))
            dudosa = (deteccion.get('uniforme_contradictorio', False) or deteccion.get('ocluido', False) or
                      (track['id'] is not None and
                       (deteccion['equipo'] != track['equipo'] or not deteccion.get('equipo_fuerte', False))))
            if track['id'] is not None and not self.memoria_confiable(track, deteccion):
                dudosa = True
            if opuesto:
                track['contrarios'] = track.get('contrarios', 0) + 1
            elif deteccion.get('equipo_fuerte', False) and deteccion['equipo'] == track['equipo']:
                track['contrarios'] = 0
            if track['id'] is not None and dudosa:
                # Actualizar geometria sin contaminar la plantilla con una oclusion.
                for clave in ('vector', 'histograma', 'partes', 'reid'):
                    muestra[clave] = anterior.get(clave, {} if clave == 'partes' else None)
            track['historia'].append(muestra)
            track['aliases'][deteccion['raw_id']] = frame
            track['aliases'] = {raw: visto for raw, visto in track['aliases'].items()
                                if frame - visto <= self.memoria}
            if not dudosa and (not track['galeria'] or frame - track['galeria'][-1]['frame'] >= 5):
                track['galeria'].append(muestra)
            numero = deteccion.get('dorsal')
            if numero is not None and (track.get('dorsal_fijo') is None or track['dorsal_fijo'] == numero):
                track['dorsales'].append(numero)
                consenso = numero if deteccion.get('dorsal_verificado', False) else self.dorsal_confirmado(track)
                if consenso is not None and track.get('dorsal_fijo') is None:
                    propietario = self.propietario_dorsal(track['equipo'], consenso)
                    if propietario is None or propietario is track:
                        track['dorsal_fijo'] = consenso
            # Ventana en frames reales, no las ultimas diez apariciones dispersas.
            while track['historia'] and track['historia'][0]['frame'] <= frame - self.ventana:
                track['historia'].popleft()
            if track['id'] is None and len(track['historia']) >= 3:
                votos = [sum(m['equipo'] == equipo for m in track['historia'])
                         for equipo in range(3)]
                equipo = int(np.argmax(votos))
                if votos[equipo] >= 3:
                    recuperado = self.recuperar(track, equipo, frame)
                    if recuperado is not None:
                        recuperado['historia'] = deque(track['historia'], maxlen=self.ventana)
                        recuperado['galeria'].extend(m for m in track['galeria']
                                                     if self.memoria_confiable(recuperado, m))
                        recuperado['dorsales'].extend(track['dorsales'])
                        if recuperado.get('dorsal_fijo') is None and track.get('dorsal_fijo') is not None:
                            recuperado['dorsal_fijo'] = track['dorsal_fijo']
                        recuperado['aliases'] = dict(track['aliases'])
                        recuperado['espacio_valido'] = True
                        track['fusionado'] = True
                        track = recuperado
                        self.recuperaciones += 1
                    cupos = (range(1, 6), range(6, 11), range(11, 13))
                    usados = {t['id'] for t in self.tracks if t['id'] is not None}
                    libre = next((k for k in cupos[equipo] if k not in usados), None)
                    esperando_reid = any(t['id'] is not None and t['equipo'] == equipo
                                         and not t.get('espacio_valido', True) for t in self.tracks)
                    if (track['id'] is None and libre is not None
                            and not esperando_reid
                            and self.propietario_dorsal(equipo, self.dorsal_confirmado(track)) is None
                            and len(track['historia']) == self.ventana and votos[equipo] >= 8):
                        track['id'], track['equipo'] = libre, equipo
            if track['id'] is not None:
                self.fijar_ancla(track)
                if (deteccion.get('dorsal_verificado', False) and numero == self.dorsal_confirmado(track)
                        and not deteccion.get('ocluido', False) and deteccion.get('reid') is not None):
                    vistas = track.setdefault('vistas_dorsal', deque(maxlen=3))
                    if all(float(v @ deteccion['reid']) < .995 for v in list(track.get('anclas', []))+list(vistas)):
                        vistas.append(deteccion['reid'].copy())
                # Agregar pocas vistas distintas solo con evidencia propia y sin rival similar.
                if (not dudosa and muestra.get('reid') is not None and len(track.get('anclas', [])) < 3
                        and self.memoria_confiable(track, muestra)
                        and all(float(v @ muestra['reid']) < .995 for v in track.get('anclas', []))):
                    track.setdefault('anclas', []).append(muestra['reid'].copy())
                if deteccion['raw_id'] >= 0:
                    self.propietarios_raw[deteccion['raw_id']] = track['id']
                visibles.append(dict(deteccion, jugador_id=track['id'],
                                     equipo=track['equipo'],
                                     dorsal=self.dorsal_confirmado(track)))
        self.tracks = [t for t in self.tracks if not t.get('fusionado', False)]
        return visibles
