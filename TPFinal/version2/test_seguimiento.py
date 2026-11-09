import unittest

import numpy as np

from seguimiento import SeguimientoJugadores


def deteccion(x=0, raw_id=1, equipo=0):
    return dict(box=np.array([x, 0, x + 30, 100], dtype=float),
                raw_id=raw_id, equipo=equipo,
                vector=np.array([1.0, 0.0]), histograma=np.array([1.0, 0.0]))


class SeguimientoTests(unittest.TestCase):
    def test_elimina_doble_recuadro_observado_sobre_azul_8(self):
        cajas = np.array([[1271.84,496.60,1392.49,728.32],
                          [1295.14,503.13,1404.20,729.04],
                          [300.,400.,350.,600.]])
        indices = SeguimientoJugadores.indices_sin_duplicados(cajas, [.9,.8,.7], [0,0,1])
        self.assertEqual(indices, [0,2])
        # Dos rivales solapados no se consideran la misma persona.
        self.assertEqual(SeguimientoJugadores.indices_sin_duplicados(cajas[:2],[.9,.8],[0,1]),[0,1])
    def test_confirma_diez_frames_y_conserva_id_si_cambia_bytetrack(self):
        tracker = SeguimientoJugadores()
        for frame in range(9):
            self.assertEqual(tracker.actualizar([deteccion(frame)], frame), [])
        visible = tracker.actualizar([deteccion(9)], 9)
        self.assertEqual(visible[0]['jugador_id'], 1)
        visible = tracker.actualizar([deteccion(10, raw_id=200)], 10)
        self.assertEqual(visible[0]['jugador_id'], 1)

    def test_recupera_occlusion_sin_crear_otro_id(self):
        tracker = SeguimientoJugadores()
        for frame in range(10):
            tracker.actualizar([deteccion()], frame)
        for frame in range(10, 20):
            self.assertEqual(tracker.actualizar([], frame), [])
        visible = tracker.actualizar([deteccion(raw_id=77)], 20)
        self.assertEqual(visible[0]['jugador_id'], 1)

    def test_cupos_y_ids_unicos(self):
        tracker = SeguimientoJugadores()
        # Seis candidatos de A, cinco de B, tres posibles arbitros.
        equipos = [0] * 6 + [1] * 5 + [2] * 3
        for frame in range(10):
            visibles = tracker.actualizar([
                deteccion(x=i * 300, raw_id=i, equipo=equipo)
                for i, equipo in enumerate(equipos)
            ], frame)
        ids = [v['jugador_id'] for v in visibles]
        self.assertEqual(sorted(ids), list(range(1, 13)))
        self.assertEqual(len(ids), len(set(ids)))

    def test_no_confirma_observaciones_separadas_por_huecos(self):
        tracker = SeguimientoJugadores()
        for frame in range(20):
            visibles = tracker.actualizar([deteccion()] if frame % 2 == 0 else [], frame)
            self.assertEqual(visibles, [])

    def test_votos_ambiguos_no_bloquean_equipo(self):
        tracker = SeguimientoJugadores()
        for frame in range(10):
            visibles = tracker.actualizar([deteccion(equipo=frame % 2)], frame)
        self.assertEqual(visibles, [])

    def test_dorsal_requiere_tres_lecturas_y_se_conserva_si_no_es_visible(self):
        tracker = SeguimientoJugadores()
        for frame in range(10):
            muestra = deteccion()
            muestra['dorsal'] = '07' if frame in (0, 3, 6) else None
            visibles = tracker.actualizar([muestra], frame)
        self.assertEqual(visibles[0]['dorsal'], '07')
        visibles = tracker.actualizar([deteccion(raw_id=99)], 10)
        self.assertEqual(visibles[0]['dorsal'], '07')

    def test_regiones_distinguen_personas_con_camiseta_igual(self):
        tracker = SeguimientoJugadores()
        for frame in range(10):
            muestra = deteccion()
            muestra['partes'] = {'shorts': np.array([1., 0.]),
                                'cabeza': np.array([1., 0.])}
            tracker.actualizar([muestra], frame)
        similar = deteccion()
        similar['partes'] = {'shorts': np.array([1., 0.]), 'cabeza': np.array([1., 0.])}
        distinto = deteccion()
        distinto['partes'] = {'shorts': np.array([0., 1.]), 'cabeza': np.array([0., 1.])}
        self.assertLess(tracker.costo(tracker.tracks[0], similar, 10),
                        tracker.costo(tracker.tracks[0], distinto, 10))

    def test_voto_equipo_ruidoso_no_rompe_id_confirmado(self):
        tracker = SeguimientoJugadores()
        for frame in range(10):
            tracker.actualizar([deteccion()], frame)
        visibles = tracker.actualizar([deteccion(equipo=1)], 10)
        self.assertEqual(visibles[0]['jugador_id'], 1)
        self.assertEqual(visibles[0]['equipo'], 0)

    def test_recupera_reservado_despues_de_memoria_con_evidencia(self):
        tracker = SeguimientoJugadores(memoria=15)
        for frame in range(10):
            muestra = deteccion()
            muestra['partes'] = {'cuerpo': np.array([1., 0.])}
            tracker.actualizar([muestra], frame)
        muestra = deteccion()
        muestra['partes'] = {'cuerpo': np.array([1., 0.])}
        visibles = tracker.actualizar([muestra], 40)
        self.assertEqual(visibles[0]['jugador_id'], 1)

    def test_aceleracion_con_bytetrack_continuo_conserva_id(self):
        tracker = SeguimientoJugadores()
        for frame in range(10):
            tracker.actualizar([deteccion()], frame)
        visibles = tracker.actualizar([deteccion(x=150)], 10)
        self.assertEqual(visibles[0]['jugador_id'], 1)

    def test_ids_sinteticos_sin_bytetrack_no_dan_bono_de_continuidad(self):
        tracker = SeguimientoJugadores()
        for frame in range(10):
            tracker.actualizar([deteccion(raw_id=-1)], frame)
        # El indice de una deteccion sin tracking no es identidad estable.
        visibles = tracker.actualizar([deteccion(x=150, raw_id=-1)], 10)
        self.assertEqual(visibles, [])

    def test_reidentifica_con_raw_nuevo_y_fuera_de_radio_espacial(self):
        tracker = SeguimientoJugadores(memoria=15)
        def muestra(x, raw):
            d = deteccion(x=x, raw_id=raw)
            d['partes'] = {'cuerpo': np.array([1., 0.]), 'shorts': np.array([1., 0.])}
            return d
        for frame in range(10):
            tracker.actualizar([muestra(0, 1)], frame)
        for frame in range(40, 50):
            visibles = tracker.actualizar([muestra(1000, 99)], frame)
        self.assertEqual(visibles[0]['jugador_id'], 1)
        self.assertEqual(tracker.recuperaciones, 1)
        self.assertEqual(sum(t['id'] is not None for t in tracker.tracks), 1)

    def test_no_recupera_si_dos_reservados_son_indistinguibles(self):
        tracker = SeguimientoJugadores(memoria=15)
        def muestra(x, raw):
            d = deteccion(x=x, raw_id=raw)
            d['partes'] = {'cuerpo': np.array([1., 0.]), 'shorts': np.array([1., 0.])}
            return d
        for frame in range(10):
            tracker.actualizar([muestra(0, 1), muestra(1000, 2)], frame)
        for frame in range(40, 50):
            tracker.actualizar([muestra(3000, 99)], frame)
        self.assertEqual(tracker.recuperaciones, 0)

    def test_color_fuerte_contrario_no_arrastra_id_por_bytetrack(self):
        tracker = SeguimientoJugadores()
        for frame in range(10):
            tracker.actualizar([deteccion()], frame)
        muestra = deteccion(equipo=1)
        muestra['equipo_fuerte'] = True
        # Con apariencia identica tolerar dos lecturas ruidosas, nunca un cambio persistente.
        self.assertEqual(tracker.actualizar([muestra], 10)[0]['equipo'], 0)
        self.assertEqual(tracker.actualizar([muestra], 11)[0]['equipo'], 0)
        self.assertEqual(tracker.actualizar([muestra], 12), [])
        visible = tracker.actualizar([deteccion()], 13)
        self.assertEqual(visible[0]['jugador_id'], 1)

    def test_compensacion_de_paneo_conserva_id_con_raw_nuevo(self):
        tracker = SeguimientoJugadores()
        for frame in range(10):
            tracker.actualizar([deteccion()], frame)
        tracker.compensar_camara(np.array([[1., 0., 1000.], [0., 1., 0.]]))
        visibles = tracker.actualizar([deteccion(x=1000, raw_id=99)], 10)
        self.assertEqual(visibles[0]['jugador_id'], 1)

    def test_uniforme_ocluido_con_continuidad_conserva_id_y_plantilla(self):
        tracker = SeguimientoJugadores()
        for frame in range(10):
            tracker.actualizar([deteccion()], frame)
        muestra = deteccion(equipo=None)
        muestra['uniforme_contradictorio'] = True
        muestra['vector'] = np.array([0., 1.])
        visible = tracker.actualizar([muestra], 10)
        self.assertEqual(visible[0]['jugador_id'], 1)
        np.testing.assert_array_equal(tracker.tracks[0]['historia'][-1]['vector'], [1., 0.])

    def test_recupera_id_con_tres_frames_sin_reconfirmar_diez(self):
        tracker = SeguimientoJugadores(memoria=15)
        def muestra(x, raw):
            d = deteccion(x=x, raw_id=raw)
            d['partes'] = {'cuerpo': np.array([1., 0.]), 'shorts': np.array([1., 0.])}
            return d
        for frame in range(10):
            tracker.actualizar([muestra(0, 1)], frame)
        self.assertEqual(tracker.actualizar([muestra(1000, 99)], 40), [])
        self.assertEqual(tracker.actualizar([muestra(1000, 99)], 41), [])
        visible = tracker.actualizar([muestra(1000, 99)], 42)
        self.assertEqual(visible[0]['jugador_id'], 1)

    def test_predicciones_son_limitadas_y_no_crean_ids(self):
        tracker = SeguimientoJugadores()
        for frame in range(10):
            tracker.actualizar([deteccion()], frame)
        for frame in range(10, 15):
            tracker.actualizar([], frame)
            pred = tracker.predecir(frame, 1920, 1080, [])
            self.assertEqual(pred[0]['jugador_id'], 1)
        self.assertEqual(tracker.predecir(15, 1920, 1080, []), [])

    def test_primer_plano_reserva_ids_sin_asociacion_espacial_ni_prediccion(self):
        tracker = SeguimientoJugadores()
        for frame in range(10):
            tracker.actualizar([deteccion()], frame)
        tracker.cambio_de_plano()
        self.assertEqual(tracker.tracks[0]['id'], 1)
        self.assertFalse(tracker.conocido(1, deteccion()['box'], 10))
        self.assertEqual(tracker.predecir(10, 1920, 1080, []), [])

    def test_raw_nuevo_cercano_puede_superar_filtro_de_piso(self):
        tracker = SeguimientoJugadores()
        for frame in range(10):
            tracker.actualizar([deteccion()], frame)
        self.assertTrue(tracker.conocido(99, deteccion(x=2)['box'], 10))
        self.assertFalse(tracker.conocido(99, deteccion(x=1000)['box'], 10))

    def test_reid_distingue_ids_perdidos_con_color_identico(self):
        tracker = SeguimientoJugadores(memoria=15)
        def muestra(x, raw, reid):
            d = deteccion(x=x, raw_id=raw)
            d['partes'] = {'cuerpo': np.array([1., 0.]), 'shorts': np.array([1., 0.])}
            d['reid'] = np.array(reid, dtype=float)
            return d
        for frame in range(10):
            tracker.actualizar([muestra(0, 1, [1, 0]), muestra(500, 2, [0, 1])], frame)
        for frame in range(40, 43):
            visibles = tracker.actualizar([muestra(2000, 99, [1, 0])], frame)
        self.assertEqual(visibles[0]['jugador_id'], 1)

    def test_reid_incompatible_veta_continuidad_raw(self):
        tracker = SeguimientoJugadores()
        for frame in range(10):
            d = deteccion()
            d['reid'] = np.array([1., 0.])
            tracker.actualizar([d], frame)
        d = deteccion()
        d['reid'] = np.array([0., 1.])
        self.assertEqual(tracker.actualizar([d], 10), [])

    def test_solapamiento_no_contamina_galeria_de_identidad(self):
        tracker = SeguimientoJugadores()
        for frame in range(10):
            tracker.actualizar([deteccion()], frame)
        d = deteccion()
        d['vector'] = np.array([0., 1.])
        d['ocluido'] = True
        d['equipo_fuerte'] = True
        visible = tracker.actualizar([d], 10)
        self.assertEqual(visible[0]['jugador_id'], 1)
        np.testing.assert_array_equal(tracker.tracks[0]['historia'][-1]['vector'], [1.,0.])


if __name__ == '__main__':
    unittest.main()
