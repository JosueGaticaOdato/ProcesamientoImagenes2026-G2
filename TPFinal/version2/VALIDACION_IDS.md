Revision visual de `tracking_mejorado.mp4`
=======================================

Se contrastaron recortes del video original con las coordenadas e IDs de
`tracking_mejorado/trayectorias.csv`. No se usaron los conteos de recuperacion
como prueba de exactitud. Los dorsales se leyeron visualmente; las muestras
sin numero legible no se anotaron como referencia.

| Equipo y dorsal | Primera muestra verificada | Muestra intermedia | Ultima muestra |
|---|---|---|---|
| A, 5 | 00:26: ID 3 | 00:56: ID 4 | 01:23 aprox.: ID 5 |
| A, 1 | 00:38: ID 5 | 00:56: ID 3 | 01:23 aprox.: ID 4 |
| B, 8 | 00:26: ID 7 | 01:14: ID 6 | 01:23 aprox.: ID 9 |
| B, 7 | 00:26: ID 9 | 01:20: ID 10 | 01:23 aprox.: ID 10 |

Los cuatro jugadores revisados cambian de ID entre las muestras. Este conjunto
es pequeno y elegido para diagnosticar errores; no permite calcular exactitud
global ni medir todos los intercambios del minuto.

En 00:38 (frame relativo 450), los IDs 1 y 3 tienen recuadros sobre el mismo
jugador azul con dorsal 8. Contar ambos como dos identidades visibles infla
el resultado. Tambien se observa un ID de arbitro sobre una persona con otra
indumentaria fuera de la cancha al final del segmento.

Evidencia: `auditoria_ids/panel_1.jpg`, `panel_2.jpg`, `panel_3.jpg`, recortes
individuales `fNNNN_idNN.jpg` y anotaciones `referencias.csv`.

Los IDs de los archivos de tracking son el resultado a evaluar, no una
referencia verdadera para validar el reconocimiento. Las referencias se
identifican por el par (equipo, dorsal).

Cambios posteriores a la auditoria
---------------------------------

Se incorporaron referencias de apariencia que no se reemplazan con muestras
dudosas, propiedad exclusiva de los IDs de ByteTrack y supresion de cajas
duplicadas del mismo equipo. El dorsal requiere tres lecturas iguales en
frames distintos; una lectura aislada no cambia un dorsal fijado.

EasyOCR se probo sobre los doce recortes anotados: siete dieron una lectura
aceptable y correcta, cinco no dieron lectura aceptable. Es un diagnostico
pequeno, no una estimacion de exactitud global ni garantia de recuperacion.
Las similitudes de `similitudes.csv` muestran que OSNet puede preferir otro
jugador del mismo equipo: bajar umbrales sin evidencia adicional no resuelve
los intercambios.

`comparar_referencias.py` permite contrastar nuevas trayectorias con estas
muestras. Distingue faltantes, duplicados, equipo incorrecto y referencias
fuera del tramo procesado. La estabilidad completa del minuto requiere
volver a comprobar esas muestras sobre la nueva salida.

Validacion de ejecucion: 44 pruebas automaticas aprobadas y un segmento de
150 frames (23-28 s) procesado sin errores, guardado en
`validacion_identidad.mp4`. En ese segmento hubo diez intentos OCR y ninguna
lectura aceptada; la consola midio aproximadamente 1.52 frames/s al frame 100.
Por ese costo, OCR queda opcional con `--verificar-dorsales`, desactivado por
defecto. Esta prueba corta comprueba ejecucion, no demuestra estabilidad de
identidad durante el minuto completo.

Comparacion con tracking_corregido (60 s, sin OCR)
------------------------------------------------

Se ejecuto `comparar_referencias.py` sobre la nueva salida y las mismas doce
muestras manuales. Detalle: `auditoria_ids/comparacion_tracking_corregido.csv`.

| Referencia | IDs en la salida anterior | IDs en la salida corregida |
|---|---|---|
| A, 5 | 3, 4, 5 | 3, 3, 4 |
| A, 1 | 5, 3, 4 | 1, 1, 5 |
| B, 8 | 7, 6, 9 | 7, 7, 7 |
| B, 7 | 9, 10, 10 | 9, 9, equipo incorrecto (ID 2) |

Hay mejora parcial, pero siguen existiendo intercambios. En B,7 la ultima
caja asociada corresponde a un ID bloqueado para A: conservar el color del
ID no garantiza que permanezca sobre la persona correcta. Esta comparacion
es puntual, no una metrica de exactitud global.

La consola del usuario termino a 3.78 frames/s, aproximadamente 7.9 minutos
para 1800 frames; la anterior termino a 4.24 frames/s. Las 110 recuperaciones
no son 110 identidades correctas. El salto de 19 a 44 entre 47 y 49 s apunta
a revisar fragmentacion antes del corte, ademas del retorno a toma abierta
en 71 s. El perfil del CSV promedia 142.9 ms en detector, 11.5 ms en camara,
74.6 ms en apariencia/OCR (OCR desactivado) y 18.1 ms en asociacion.

La proxima validacion debe centrarse en esos dos tramos usando el estado
acumulado desde 23 s; arrancar directamente despues del corte no reproduce
la recuperacion de las identidades originales.
