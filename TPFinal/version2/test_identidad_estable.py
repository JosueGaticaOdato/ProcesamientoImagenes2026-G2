import unittest

import numpy as np

from seguimiento import SeguimientoJugadores


def deteccion(x=0, raw=1, equipo=0, reid=(1.,0.), numero=None, verificado=False):
    return dict(box=np.array([x,0.,x+30.,100.]),raw_id=raw,equipo=equipo,equipo_fuerte=True,
                vector=np.array([1.,0.]),histograma=np.array([1.,0.]),
                partes={},reid=np.array(reid),dorsal=numero,dorsal_verificado=verificado)


class IdentidadEstableTests(unittest.TestCase):
    def test_un_raw_no_pertenece_a_dos_ids(self):
        s=SeguimientoJugadores()
        for f in range(10):
            s.actualizar([deteccion(),deteccion(500,2,reid=(0.,1.))],f)
        visibles=s.actualizar([deteccion(500,1,reid=(0.,1.))],10)
        self.assertEqual(visibles[0]['jugador_id'],2)
        uno=next(t for t in s.tracks if t['id']==1)
        dos=next(t for t in s.tracks if t['id']==2)
        self.assertFalse(s.raw_coincide(uno,1,10))
        self.assertTrue(s.raw_coincide(dos,1,10))

    def test_memoria_no_adopta_otro_jugador_gradualmente(self):
        s=SeguimientoJugadores()
        for f in range(10):
            s.actualizar([deteccion()],f)
        ancla=s.tracks[0]['anclas'][0].copy()
        for f in range(10,60):
            s.actualizar([deteccion(reid=(.7,np.sqrt(.51)))],f)
        np.testing.assert_array_equal(s.tracks[0]['anclas'][0],ancla)
        self.assertAlmostEqual(s.similitud_reid(s.tracks[0],{'reid':np.array([0.,1.])}),0.)

    def test_numero_fijo_no_cambia_por_lectura_aislada(self):
        s=SeguimientoJugadores()
        for f in range(10):
            s.actualizar([deteccion(numero='5',verificado=True)],f)
        s.actualizar([deteccion(numero='8')],10)
        self.assertEqual(s.dorsal_confirmado(s.tracks[0]),'5')

    def test_numero_consensuado_impide_intercambio_aunque_raw_coincida(self):
        s=SeguimientoJugadores()
        for f in range(10):
            s.actualizar([deteccion(numero='5',verificado=True),
                          deteccion(500,2,reid=(0.,1.),numero='8',verificado=True)],f)
        actual=deteccion(500,1,reid=(0.,1.),numero='8',verificado=True)
        visibles=s.actualizar([actual],10)
        self.assertEqual(visibles[0]['jugador_id'],2)
        self.assertEqual(s.dorsal_confirmado(next(t for t in s.tracks if t['id']==1)),'5')

    def test_numero_recupera_tras_corte_sin_mezclar_coordenadas_viejas(self):
        s=SeguimientoJugadores()
        for f in range(10):
            s.actualizar([deteccion(numero='5',verificado=True)],f)
        s.cambio_de_plano()
        visibles=s.actualizar([deteccion(1000,99,numero='5',verificado=True)],30)
        self.assertEqual(visibles[0]['jugador_id'],1)
        self.assertEqual(len(s.tracks[0]['historia']),1)
        self.assertTrue(s.tracks[0]['espacio_valido'])

    def test_numero_igual_en_equipos_distintos_no_es_misma_identidad(self):
        s=SeguimientoJugadores()
        for f in range(10):
            s.actualizar([deteccion(numero='8',verificado=True),
                          deteccion(500,2,equipo=1,reid=(0.,1.),numero='8',verificado=True)],f)
        visibles=s.actualizar([deteccion(0,99,equipo=1,reid=(0.,1.),numero='8',verificado=True)],10)
        self.assertEqual(visibles[0]['equipo'],1)
        self.assertEqual(visibles[0]['jugador_id'],6)

    def test_nueva_vista_aprendida_solo_con_dorsal_confirmado(self):
        s=SeguimientoJugadores()
        for f in range(10):
            s.actualizar([deteccion(numero='5',verificado=True)],f)
        nueva=deteccion(reid=(.7,np.sqrt(.51)),numero='5',verificado=True)
        s.actualizar([nueva],10)
        self.assertEqual(len(s.tracks[0]['vistas_dorsal']),1)
        self.assertAlmostEqual(s.similitud_reid(s.tracks[0],nueva),1.)


if __name__=='__main__':
    unittest.main()
