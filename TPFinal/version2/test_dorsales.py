import unittest

from dorsales import ControlDorsales
from apariencia import LectorDorsal


class DorsalesTests(unittest.TestCase):
    def test_tres_observaciones_y_no_tres_lecturas_del_mismo_frame(self):
        control=ControlDorsales()
        for _ in range(3):
            control.registrar(1,'05',0)
        self.assertIsNone(control.confirmado(1,0))
        control.registrar(1,'05',10)
        control.registrar(1,'05',20)
        self.assertEqual(control.confirmado(1,20),'05')
        self.assertIsNone(control.confirmado(1,24,reciente=3))

    def test_cambio_fisico_del_raw_no_arrastra_dorsal_viejo(self):
        control=ControlDorsales()
        for frame in (0,10,20):
            control.registrar(7,'5',frame)
        control.registrar(7,'8',30)
        self.assertIsNone(control.confirmado(7,30))
        control.registrar(7,'8',40)
        control.registrar(7,'8',50)
        self.assertEqual(control.confirmado(7,50),'8')
        self.assertIsNone(control.confirmado(7,250))

    def test_prioriza_completar_consenso_sin_que_el_tiempo_lo_rompa(self):
        control=ControlDorsales()
        control.registrar(1,'8',1000)
        self.assertGreater(control.prioridad(1,1001),control.prioridad(2,1001))

    def test_texto_pequeno_incierto_o_ambiguo_no_es_numero(self):
        caja=[[0,0],[30,0],[30,25],[0,25]]
        pequena=[[0,0],[30,0],[30,10],[0,10]]
        self.assertEqual(LectorDorsal.filtrar_lecturas([(caja,'8',.99)],100),'8')
        self.assertIsNone(LectorDorsal.filtrar_lecturas([(caja,'8',.7)],100))
        self.assertIsNone(LectorDorsal.filtrar_lecturas([(pequena,'8',.99)],100))
        self.assertIsNone(LectorDorsal.filtrar_lecturas([(caja,'363',.99)],100))
        self.assertIsNone(LectorDorsal.filtrar_lecturas([(caja,'8',.99),(caja,'7',.99)],100))


if __name__=='__main__':
    unittest.main()
