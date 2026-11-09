"""Consenso temporal de lecturas OCR por trayectoria, independiente del ID asignado."""
from collections import deque


class ControlDorsales:
    def __init__(self, max_edad=150, separacion=3):
        self.max_edad = max_edad
        self.separacion = separacion
        self.lecturas = {}
        self.intentos = {}

    def registrar(self, raw_id, numero, frame):
        self.intentos[raw_id] = frame
        if raw_id < 0 or numero is None or not numero.isdigit() or not 1 <= len(numero) <= 2:
            return
        historia = self.lecturas.setdefault(raw_id, deque(maxlen=5))
        if historia and frame - historia[-1][0] < self.separacion:
            return
        while historia and frame - historia[0][0] > self.max_edad:
            historia.popleft()
        historia.append((frame, numero))

    def confirmado(self, raw_id, frame, reciente=None):
        historia = list(self.lecturas.get(raw_id, []))
        vigentes = [(f,n) for f,n in historia if frame-f <= self.max_edad]
        if len(vigentes) < 3:
            return None
        ultimas = vigentes[-3:]
        if len({n for _,n in ultimas}) != 1:
            return None
        if reciente is not None and frame-ultimas[-1][0] > reciente:
            return None
        return ultimas[-1][1]

    def prioridad(self, raw_id, frame):
        historia = list(self.lecturas.get(raw_id, []))
        pendiente = 0
        if historia and frame-historia[-1][0] <= self.max_edad//2:
            ultimo = historia[-1][1]
            for _, n in reversed(historia):
                if n != ultimo:
                    break
                pendiente += 1
        bonus = 1000 if pendiente == 2 else 500 if pendiente == 1 else 0
        edad = (self.max_edad*2 if raw_id not in self.intentos
                else min(self.max_edad, frame-self.intentos[raw_id]))
        return bonus+edad

    def limpiar(self, frame):
        self.lecturas = {raw:h for raw,h in self.lecturas.items()
                         if h and frame-h[-1][0] <= self.max_edad}
        self.intentos = {raw:f for raw,f in self.intentos.items() if frame-f <= self.max_edad}
