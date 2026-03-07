# src/simulation_ecuador_completo.py
"""
Simulación realista de demanda para farmacias en Ecuador
Con catálogo realista de 10,000 SKUs - VERSIÓN BIG DATA
"""

import glob
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import pyarrow
import fastparquet
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

class CategoriasFarmaceuticasEcuador:
    """
    Categorización profesional basada en:
    1. Clasificación ATC (Anatomical Therapeutic Chemical)
    2. Plan Básico de Medicamentos MSP Ecuador
    3. Categorización de CADFAR (Cámara de Distribuidores de Farmacias)
    """

    @staticmethod
    def get_categories():
        """Retorna categorías farmacéuticas completas para Ecuador"""
        return {
            # GRUPO A: SISTEMA DIGESTIVO Y METABOLISMO
            "A01": "Estomatológicos y Antisépticos Bucales",
            "A02": "Antiácidos y Antidislipémicos",
            "A03": "Antiespasmódicos y Anticolinérgicos",
            "A04": "Antieméticos y Antinauseosos",
            "A05": "Terapia Biliar y Hepática",
            "A06": "Laxantes y Antidiarreicos",
            "A07": "Antidiarréicos y Antiinflamatorios Intestinales",
            "A08": "Antiobesidad y Nutrición",
            "A09": "Digestivos y Enzimas",
            "A10": "Antidiabéticos y Reguladores Metabólicos",
            "A11": "Vitaminas y Minerales",
            "A12": "Suplementos Minerales",
            "A13": "Tónicos y Reconstituyentes",
            "A14": "Anabólicos y Estimulantes del Apetito",
            "A15": "Estimulantes del Apetito",
            "A16": "Otros Productos Digestivos y Metabólicos",
            
            # GRUPO B: SANGRE Y ÓRGANOS HEMATOPOYÉTICOS
            "B01": "Antitrombóticos y Antiagregantes",
            "B02": "Antifibrinolíticos y Hemostáticos",
            "B03": "Antianémicos y Hematínicos",
            "B05": "Soluciones para Hemodiálisis y Diuresis",
            "B06": "Otros Agentes Hematológicos",
            
            # GRUPO C: SISTEMA CARDIOVASCULAR
            "C01": "Cardiotónicos y Antiarrítmicos",
            "C02": "Antihipertensivos",
            "C03": "Diuréticos",
            "C04": "Vasodilatadores Periféricos",
            "C05": "Vasoprotectores y Flebotónicos",
            "C07": "Betabloqueantes",
            "C08": "Bloqueantes de Canales de Calcio",
            "C09": "Inhibidores de la ECA y ARA II",
            "C10": "Hipolipemiantes y Antidisplipémicos",
            
            # GRUPO D: DERMATOLÓGICOS
            "D01": "Antifúngicos Dermatológicos",
            "D02": "Emolientes y Protectores",
            "D03": "Cicatrizantes y Repitelizantes",
            "D04": "Antipruriginosos y Anestésicos Locales",
            "D05": "Antipsoriásicos y Queratolíticos",
            "D06": "Antibióticos y Quimioterápicos Dermatológicos",
            "D07": "Corticosteroides Dermatológicos",
            "D08": "Antisépticos y Desinfectantes",
            "D09": "Apósitos Medicados",
            "D10": "Preparados Antiacné",
            "D11": "Otros Preparados Dermatológicos",
            
            # GRUPO G: SISTEMA GENITOURINARIO Y HORMONAS SEXUALES
            "G01": "Antinfecciosos y Antisépticos Ginecológicos",
            "G02": "Otros Agentes Ginecológicos",
            "G03": "Hormonas Sexuales y Moduladores del Sistema Genital",
            "G04": "Preparados Urológicos",
            
            # GRUPO H: PREPARADOS HORMONALES SISTÉMICOS (excluyendo sexuales)
            "H01": "Hormonas Hipofisarias e Hipotalámicas",
            "H02": "Corticosteroides Sistémicos",
            "H03": "Terapia Tiroidea y Antitiroidea",
            "H04": "Hormonas Pancreáticas",
            "H05": "Reguladores del Metabolismo del Calcio",
            
            # GRUPO J: ANTIINFECCIOSOS SISTÉMICOS
            "J01": "Antibacterianos Sistémicos",
            "J02": "Antimicóticos Sistémicos",
            "J04": "Antimicobacterianos",
            "J05": "Antivirales Sistémicos",
            "J06": "Inmunosueros e Inmunoglobulinas",
            "J07": "Vacunas",
            
            # GRUPO L: AGENTES ANTINEOPLÁSICOS E INMUNOMODULADORES
            "L01": "Agentes Antineoplásicos",
            "L02": "Agentes Endocrinos Antineoplásicos",
            "L03": "Inmunoestimulantes",
            "L04": "Inmunosupresores",
            
            # GRUPO M: SISTEMA MUSCULOESQUELÉTICO
            "M01": "Antiinflamatorios y Antirreumáticos",
            "M02": "Analgésicos Tópicos",
            "M03": "Relajantes Musculares",
            "M04": "Antigotosos",
            "M05": "Fármacos para Enfermedades Óseas",
            "M09": "Otros Fármacos para Trastornos Musculoesqueléticos",
            
            # GRUPO N: SISTEMA NERVIOSO
            "N01": "Anestésicos",
            "N02": "Analgésicos y Antipiréticos",
            "N03": "Antiepilépticos",
            "N04": "Antiparkinsonianos",
            "N05": "Psicolepticos (Antipsicóticos, etc.)",
            "N06": "Psicoanalépticos (Antidepresivos, etc.)",
            "N07": "Otros Fármacos del Sistema Nervioso",
            
            # GRUPO P: ANTIPARASITARIOS, INSECTICIDAS Y REPELENTES
            "P01": "Antiprotozoarios",
            "P02": "Antihelmínticos",
            "P03": "Ectoparasiticidas",
            
            # GRUPO R: SISTEMA RESPIRATORIO
            "R01": "Preparados Nasales",
            "R02": "Preparados para la Garganta",
            "R03": "Antiasmáticos y Broncodilatadores",
            "R05": "Preparados para la Tos y el Resfriado",
            "R06": "Antihistamínicos Sistémicos",
            "R07": "Otros Fármacos del Sistema Respiratorio",
            
            # GRUPO S: ÓRGANOS DE LOS SENTIDOS
            "S01": "Oftalmológicos",
            "S02": "Otológicos",
            "S03": "Oftalmológicos y Otológicos Combinados",
            
            # GRUPO V: VARIOS
            "V01": "Alérgenos",
            "V03": "Todos los Otros Productos Terapéuticos",
            "V04": "Agentes Diagnósticos",
            "V06": "Nutrición General",
            "V07": "Todos los Otros Productos no Terapéuticos",
            "V08": "Medios de Contraste",
            "V09": "Radiofármacos Diagnósticos",
            "V10": "Radiofármacos Terapéuticos",
            "V20": "Apósitos Quirúrgicos"
        }
    
    @staticmethod
    def get_simplified_categories():
        """Versión simplificada para simulación"""
        return [
            # 1. MEDICAMENTOS ÉTICOS (RECETA MÉDICA) - 40% del catálogo ~6,800 SKUs
            "Antibióticos Sistémicos",
            "Antihipertensivos y Cardiotónicos", 
            "Antidiabéticos Orales e Insulina",
            "Antiasmáticos y Broncodilatadores",
            "Antidepresivos y Ansiolíticos",
            "Antipsicóticos y Estabilizadores del Ánimo",
            "Anticonceptivos Hormonales",
            "Corticosteroides Sistémicos",
            "Antiepilépticos y Antiparkinsonianos",
            "Antineoplásicos e Inmunomoduladores",
            "Antiarrítmicos",
            "Anticoagulantes",
            "Hormonas Tiroideas",
            "Inmunosupresores",
            "Antirretrovirales",
            "Antifúngicos Sistémicos",
            
            # 2. MEDICAMENTOS OTC (VENTA LIBRE) - 25% ~4,250 SKUs
            "Analgésicos y Antiinflamatorios OTC",
            "Antigripales y Antitusivos",
            "Antialérgicos y Antihistamínicos OTC",
            "Antiácidos y Digestivos",
            "Laxantes y Antidiarreicos",
            "Vitaminas y Suplementos Alimenticios",
            "Suplementos Minerales (Hierro, Calcio, etc.)",
            "Tónicos y Reconstituyentes",
            "Antimicóticos Tópicos OTC",
            "Analgésicos Tópicos",
            
            # 3. DERMOCOSMÉTICA Y CUIDADO PERSONAL - 15% ~2,550 SKUs
            "Dermocosméticos Faciales (Antiedad, Hidratantes)",
            "Productos para el Acné y Dermatitis",
            "Protectores Solares y Post-solares",
            "Cremas Corporales y Emolientes",
            "Repelentes de Insectos",
            "Cuidado Capilar Especializado",
            "Antitranspirantes y Desodorantes",
            "Cuidado Íntimo Femenino",
            "Jabones Dermatológicos",
            "Productos para Dermatitis Atópica",
            "Cosmética Natural",
            
            # 4. MATERIALES E INSUMOS MÉDICOS - 10% ~1,700 SKUs
            "Material de Curación (Gasas, Apósitos, Vendajes)",
            "Equipos de Diagnóstico (Termómetros, Tensiómetros, Glucómetros)",
            "Ayudas Ortopédicas (Rodilleras, Tobilleras, Fajas)",
            "Incontinencia y Geriatría (Pañales, Protectores)",
            "Cuidado de Heridas y Cicatrizantes",
            "Suplementos Nutricionales Enterales",
            "Jeringuillas y Agujas",
            "Equipos de Infusión",
            
            # 5. PRODUCTOS ESPECIALIZADOS - 7% ~1,200 SKUs
            "Pediatría y Lactantes (Leches, Papillas, Accesorios)",
            "Maternidad y Prenatal (Ácido Fólico, Suplementos)",
            "Salud Sexual (Preservativos, Lubricantes)",
            "Cuidado Visual (Lentillas, Soluciones)",
            "Salud Bucal (Pastas, Enjuagues, Cepillos Especiales)",
            "Homeopatía y Fitoterapia",
            "Productos Naturales",
            "Aceites Esenciales",
            
            # 6. EQUIPOS MÉDICOS (DISPOSITIVOS MÉDICOS) - 3% ~500 SKUs
            "Monitores de Salud (Tensiómetros Digitales)",
            "Nebulizadores y Aspiradores Nasales",
            "Oxímetros de Pulso",
            "Sillas de Ruedas y Andaderas",
            "Colchones y Cojines Antiescaras",
            "Equipos de Inhaloterapia",
            "Glucómetros y Tiras Reactivas",
            "Tensiómetros de Brazo",
            "Termómetros Digitales e Infrarrojos",
            
            # 7. GENÉRICOS (POR PRINCIPIO ACTIVO) - 10% ~1,700 SKUs
            "Genéricos Cardiovasculares",
            "Genéricos del Sistema Nervioso",
            "Genéricos Antiinfecciosos",
            "Genéricos Digestivos y Metabólicos",
            "Genéricos Respiratorios",
            "Genéricos Dermatológicos",
            "Genéricos Endocrinológicos"
        ]

class EcuadorSeasonalityAdjusted:
    """Factores estacionales ajustados para categorías farmacéuticas"""
    
    @staticmethod
    def get_category_factors(categoria, month):
        """Factores específicos por categoría y mes en Ecuador"""
        
        factores = {
            # ANTIBIÓTICOS: Picos en temporada de lluvias
            "Antibióticos Sistémicos": {
                1: 1.0, 2: 1.1, 3: 1.3, 4: 1.4, 5: 1.2, 6: 1.1,
                7: 1.0, 8: 1.0, 9: 1.1, 10: 1.2, 11: 1.3, 12: 1.4
            },
            
            # ANTIGRIPALES: Invierno en Sierra (jun-ago) y cambios de estación
            "Antigripales y Antitusivos": {
                1: 1.1, 2: 1.2, 3: 1.1, 4: 1.0, 5: 1.0, 6: 1.3,
                7: 1.5, 8: 1.4, 9: 1.2, 10: 1.1, 11: 1.2, 12: 1.4
            },
            
            # VITAMINAS: Propósitos año nuevo y preparación invierno
            "Vitaminas y Suplementos Alimenticios": {
                1: 1.6, 2: 1.4, 3: 1.2, 4: 1.1, 5: 1.3, 6: 1.2,
                7: 1.1, 8: 1.1, 9: 1.2, 10: 1.4, 11: 1.5, 12: 1.8
            },
            
            # DERMOCOSMÉTICA: Verano y Día de la Madre
            "Dermocosméticos Faciales": {
                1: 1.5, 2: 1.6, 3: 1.3, 4: 1.2, 5: 1.8, 6: 1.4,
                7: 1.3, 8: 1.2, 9: 1.3, 10: 1.4, 11: 1.5, 12: 1.7
            },
            
            # PROTECTORES SOLARES: Verano ecuatorial
            "Protectores Solares y Post-solares": {
                1: 1.8, 2: 1.9, 3: 1.7, 4: 1.5, 5: 1.4, 6: 1.3,
                7: 1.2, 8: 1.3, 9: 1.4, 10: 1.5, 11: 1.6, 12: 1.7
            },
            
            # REPELENTES: Temporada de lluvias (mosquitos)
            "Repelentes de Insectos": {
                1: 1.2, 2: 1.3, 3: 1.5, 4: 1.7, 5: 1.6, 6: 1.4,
                7: 1.3, 8: 1.2, 9: 1.3, 10: 1.4, 11: 1.3, 12: 1.2
            },
            
            # ANTIÁCIDOS: Fiestas y excesos alimenticios
            "Antiácidos y Digestivos": {
                1: 1.4, 2: 1.5, 3: 1.3, 4: 1.6, 5: 1.3, 6: 1.2,
                7: 1.1, 8: 1.1, 9: 1.2, 10: 1.7, 11: 1.5, 12: 1.8
            },
            
            # MATERNIDAD: Distribución uniforme con picos post-fiestas
            "Maternidad y Prenatal": {
                1: 1.2, 2: 1.1, 3: 1.1, 4: 1.2, 5: 1.3, 6: 1.2,
                7: 1.1, 8: 1.3, 9: 1.4, 10: 1.3, 11: 1.2, 12: 1.4
            },
            
            # PEDIATRÍA: Todo el año, pico en Día del Niño
            "Pediatría y Lactantes": {
                1: 1.3, 2: 1.2, 3: 1.1, 4: 1.2, 5: 1.3, 6: 1.7,
                7: 1.2, 8: 1.1, 9: 1.2, 10: 1.3, 11: 1.2, 12: 1.5
            },
            
            # EQUIPOS MÉDICOS: Compras planificadas, pico fin de año
            "Monitores de Salud (Tensiómetros Digitales)": {
                1: 1.4, 2: 1.2, 3: 1.1, 4: 1.0, 5: 1.1, 6: 1.0,
                7: 1.0, 8: 1.0, 9: 1.1, 10: 1.2, 11: 1.4, 12: 1.6
            },
            
            # GENÉRICOS: Demanda constante con leve aumento fin de mes
            "Genéricos Cardiovasculares": {
                1: 1.1, 2: 1.1, 3: 1.0, 4: 1.1, 5: 1.0, 6: 1.1,
                7: 1.0, 8: 1.1, 9: 1.0, 10: 1.1, 11: 1.2, 12: 1.3
            }
        }
        
        # Si la categoría tiene factores específicos, usarlos
        if categoria in factores:
            return factores[categoria].get(month, 1.0)
        
        # Para categorías sin factores específicos, usar patrón general
        # Picos en: Enero (propósitos), Mayo (Día Madre), Octubre (Fiestas), Diciembre (Navidad)
        picos = {1: 1.3, 5: 1.4, 10: 1.5, 12: 1.6}
        return picos.get(month, 1.0)
    
    @staticmethod
    def get_demand_distribution(categoria):
        """Distribución de demanda por categoría"""
        distribuciones = {
            # Medicamentos crónicos: demanda constante, baja variación
            "Antihipertensivos y Cardiotónicos": ("normal", 25, 5),
            "Antidiabéticos Orales e Insulina": ("normal", 20, 4),
            "Antidepresivos y Ansiolíticos": ("normal", 18, 3),
            
            # Medicamentos agudos: Poisson (eventos independientes)
            "Antibióticos Sistémicos": ("poisson", 22),
            "Antigripales y Antitusivos": ("poisson", 35),
            "Analgésicos y Antiinflamatorios OTC": ("poisson", 45),
            
            # Suplementos: Binomial Negativa (demanda irregular)
            "Vitaminas y Suplementos Alimenticios": ("negative_binomial", 12, 0.4),
            "Suplementos Minerales": ("negative_binomial", 8, 0.35),
            
            # Dermocosméticos: Lognormal (algunos productos muy populares)
            "Dermocosméticos Faciales": ("lognormal", 2.5, 0.5),
            "Protectores Solares y Post-solares": ("lognormal", 2.8, 0.6),
            
            # Equipos médicos: Poisson baja frecuencia
            "Monitores de Salud": ("poisson", 3),
            "Nebulizadores y Aspiradores Nasales": ("poisson", 2),
            
            # Materiales: Normal con variación media
            "Material de Curación": ("normal", 30, 8),
            "Incontinencia y Geriatría": ("normal", 25, 6),
            
            # Genéricos: Poisson con frecuencia media
            "Genéricos Cardiovasculares": ("poisson", 28),
            "Genéricos del Sistema Nervioso": ("poisson", 22),
        }
        
        return distribuciones.get(categoria, ("poisson", 20))

class EcuadorPharmaSimulator:
    """Simulador con catálogo masivo de 10,000 SKUs - VERSIÓN BIG DATA"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)

        # Nuevos parámetros
        self.total_skus_catalogo = 10000  # 8-10 mil SKUs totales
        self.skus_por_farmacia_min = 2000
        self.skus_por_farmacia_max = 4000
        
        self.categorias = CategoriasFarmaceuticasEcuador.get_simplified_categories()
        self.regiones = ["Sierra", "Costa", "Oriente", "Insular"]
        self.ciudades = {
            "Sierra": ["Quito", "Cuenca", "Ambato", "Riobamba", "Latacunga", "Ibarra", "Loja"],
            "Costa": ["Guayaquil", "Manta", "Machala", "Esmeraldas", "Portoviejo", "Santo Domingo"],
            "Oriente": ["Puyo", "Tena", "Macas", "Coca", "Lago Agrio"],
            "Insular": ["Puerto Ayora", "Puerto Baquerizo", "Puerto Villamil"]
        }
        
        # GENERAR CATÁLOGO MASIVO DE 17,000 SKUs
        self.productos_catalogo = self._generar_catalogo_masivo()
        
        # Guardar catálogo por separado (siempre útil)
        self.productos_catalogo.to_csv('data/catalogo_productos_10000.csv', index=False)

        # ===== NUEVO: Evento atípico para simular disrupción =====
        self.EVENTO_ATIPICO = {
            'nombre': 'Crisis sanitaria regional',
            'semanas_afectadas': list(range(31, 43)),  # Semanas 31 a 42 (12 semanas)
            'descripcion': 'Restricciones a importaciones de productos farmacéuticos',
            'efecto': 'Imposibilidad de generar órdenes de reposición',
            'factor_demanda': 1.8,  # Aumento de demanda por pánico/compras de emergencia
            'factor_importacion': 0.0  # 0% de órdenes pueden ejecutarse
        }

    # ============= MÉTODOS DE GENERACIÓN DE CATÁLOGO =============
    def _generar_catalogo_masivo(self):
        # Genera catálogo realista de 10,000 SKUs con distribución por categoría

        # Reescalar distribución anterior a 10,000 SKUs
        factor_escala = 10000 / 17025  # ≈ 0.587
        
        # Distribución de SKUs por categoría (SIN DUPLICADOS)
        distribucion_skus = {
            # Éticos (40%)
            "Antibióticos Sistémicos": int(800 * factor_escala),
            "Antihipertensivos y Cardiotónicos": int(700 * factor_escala),
            "Antidiabéticos Orales e Insulina": int(600 * factor_escala),
            "Antiasmáticos y Broncodilatadores": int(500 * factor_escala),
            "Antidepresivos y Ansiolíticos": int(600 * factor_escala),
            "Antipsicóticos y Estabilizadores del Ánimo": int(500 * factor_escala),
            "Anticonceptivos Hormonales": int(400 * factor_escala),
            "Corticosteroides Sistémicos": int(400 * factor_escala),
            "Antiepilépticos y Antiparkinsonianos": int(450 * factor_escala),
            "Antineoplásicos e Inmunomoduladores": int(600 * factor_escala),
            "Antiarrítmicos": int(300 * factor_escala),
            "Anticoagulantes": int(300 * factor_escala),
            "Hormonas Tiroideas": int(250 * factor_escala),
            "Inmunosupresores": int(350 * factor_escala),
            "Antirretrovirales": int(400 * factor_escala),
            "Antifúngicos Sistémicos": int(300 * factor_escala),
            
            # OTC (25%)
            "Analgésicos y Antiinflamatorios OTC": int(800 * factor_escala),
            "Antigripales y Antitusivos": int(700 * factor_escala),
            "Antialérgicos y Antihistamínicos OTC": int(500 * factor_escala),
            "Antiácidos y Digestivos": int(500 * factor_escala),
            "Laxantes y Antidiarreicos": int(400 * factor_escala),
            "Vitaminas y Suplementos Alimenticios": int(700 * factor_escala),
            "Suplementos Minerales (Hierro, Calcio, etc.)": int(400 * factor_escala),
            "Tónicos y Reconstituyentes": int(250 * factor_escala),
            "Antimicóticos Tópicos OTC": int(300 * factor_escala),
            "Analgésicos Tópicos": int(300 * factor_escala),
            
            # Dermocosméticos (15%)
            "Dermocosméticos Faciales (Antiedad, Hidratantes)": int(500 * factor_escala),
            "Productos para el Acné y Dermatitis": int(350 * factor_escala),
            "Protectores Solares y Post-solares": int(300 * factor_escala),
            "Cremas Corporales y Emolientes": int(400 * factor_escala),
            "Repelentes de Insectos": int(250 * factor_escala),
            "Cuidado Capilar Especializado": int(300 * factor_escala),
            "Antitranspirantes y Desodorantes": int(200 * factor_escala),
            "Cuidado Íntimo Femenino": int(150 * factor_escala),
            "Jabones Dermatológicos": int(150 * factor_escala),
            "Productos para Dermatitis Atópica": int(150 * factor_escala),
            "Cosmética Natural": int(200 * factor_escala),
            
            # Materiales e Insumos (10%)
            "Material de Curación (Gasas, Apósitos, Vendajes)": int(500 * factor_escala),
            "Equipos de Diagnóstico (Termómetros, Tensiómetros, Glucómetros)": int(300 * factor_escala),
            "Ayudas Ortopédicas (Rodilleras, Tobilleras, Fajas)": int(250 * factor_escala),
            "Incontinencia y Geriatría (Pañales, Protectores)": int(200 * factor_escala),
            "Cuidado de Heridas y Cicatrizantes": int(200 * factor_escala),
            "Suplementos Nutricionales Enterales": int(150 * factor_escala),
            "Jeringuillas y Agujas": int(150 * factor_escala),
            "Equipos de Infusión": int(150 * factor_escala),
            
            # Especializados (7%)
            "Pediatría y Lactantes (Leches, Papillas, Accesorios)": int(300 * factor_escala),
            "Maternidad y Prenatal (Ácido Fólico, Suplementos)": int(200 * factor_escala),
            "Salud Sexual (Preservativos, Lubricantes)": int(200 * factor_escala),
            "Cuidado Visual (Lentillas, Soluciones)": int(200 * factor_escala),
            "Salud Bucal (Pastas, Enjuagues, Cepillos Especiales)": int(200 * factor_escala),
            "Homeopatía y Fitoterapia": int(150 * factor_escala),
            "Productos Naturales": int(150 * factor_escala),
            "Aceites Esenciales": int(150 * factor_escala),
            
            # Equipos Médicos (3%)
            "Monitores de Salud (Tensiómetros Digitales)": int(100 * factor_escala),
            "Nebulizadores y Aspiradores Nasales": int(80 * factor_escala),
            "Oxímetros de Pulso": int(70 * factor_escala),
            "Sillas de Ruedas y Andaderas": int(60 * factor_escala),
            "Colchones y Cojines Antiescaras": int(50 * factor_escala),
            "Equipos de Inhaloterapia": int(50 * factor_escala),
            "Glucómetros y Tiras Reactivas": int(150 * factor_escala),
            "Tensiómetros de Brazo": int(80 * factor_escala),      # <-- SOLO UNA VEZ
            "Termómetros Digitales e Infrarrojos": int(60 * factor_escala),  # <-- SOLO UNA VEZ
            
            # Genéricos (10%)
            "Genéricos Cardiovasculares": int(400 * factor_escala),
            "Genéricos del Sistema Nervioso": int(350 * factor_escala),  # <-- SOLO UNA VEZ
            "Genéricos Antiinfecciosos": int(300 * factor_escala),
            "Genéricos Digestivos y Metabólicos": int(300 * factor_escala),
            "Genéricos Respiratorios": int(250 * factor_escala),
            "Genéricos Dermatológicos": int(200 * factor_escala),
            "Genéricos Endocrinológicos": int(200 * factor_escala)
        }
        
        # Ajustar para que sume exactamente 10,000
        total_actual = sum(distribucion_skus.values())
        diferencia = 10000 - total_actual
        
        # Distribuir la diferencia en categorías principales
        categorias_principales = ["Analgésicos y Antiinflamatorios OTC", 
                                "Vitaminas y Suplementos Alimenticios",
                                "Antibióticos Sistémicos"]
        for cat in categorias_principales[:diferencia]:
            distribucion_skus[cat] += 1

        # Verificar que no hay duplicados
        print(f"\n🔍 Verificando {len(distribucion_skus)} categorías...")
        
        # Laboratorios por categoría
        laboratorios_por_categoria = {
            "Éticos": ["Pfizer", "Novartis", "Roche", "Merck", "Sanofi", "AstraZeneca", "GSK", "Bayer", "Abbott", "Boehringer"],
            "OTC": ["Bayer", "GSK", "Johnson & Johnson", "Sanofi", "Reckitt", "Procter & Gamble"],
            "Dermocosméticos": ["L'Oréal", "CeraVe", "La Roche-Posay", "Vichy", "Eucerin", "Avene", "Isdin"],
            "Genéricos": ["Mega Labs", "Lab. Chile", "Lab. Bagó", "Lab. Lafrancol", "Lab. Pharmainvesti", "Lab. Sanut", "Lab. Tecnofarma"],
            "Nacionales": ["Ecuaquímica", "Laboratorios Life", "Laboratorios Sierra", "Farmadonal", "Medipharma"]
        }
        
        productos = []
        sku_counter = 10000
        
        print("\n🏭 Generando catálogo de 10,000 SKUs...")
        
        # Crear un conjunto para rastrear categorías ya procesadas
        categorias_procesadas = set()
        
        for categoria, num_skus in distribucion_skus.items():
            # Verificar duplicados en tiempo real
            if categoria in categorias_procesadas:
                print(f"   ⚠️ ADVERTENCIA: Categoría duplicada encontrada: {categoria}")
                continue
            
            categorias_procesadas.add(categoria)
            print(f"   • {categoria[:40]:40} → {num_skus:4d} SKUs")
            
            # Determinar tipo de categoría para laboratorios
            if any(x in categoria for x in ["Antibióticos", "Antihipertensivos", "Antidiabéticos", "Antineoplásicos"]):
                labs = laboratorios_por_categoria["Éticos"]
            elif any(x in categoria for x in ["OTC", "Analgésicos", "Antigripales", "Vitaminas"]):
                labs = laboratorios_por_categoria["OTC"]
            elif any(x in categoria for x in ["Dermocosméticos", "Protectores", "Cremas"]):
                labs = laboratorios_por_categoria["Dermocosméticos"]
            elif "Genérico" in categoria:
                labs = laboratorios_por_categoria["Genéricos"]
            else:
                labs = laboratorios_por_categoria["Nacionales"]
            
            for i in range(num_skus):
                sku_id = f"SKU{sku_counter:05d}"
                
                # Generar nombre de producto
                if "Genérico" in categoria:
                    nombre = self._generar_nombre_generico(categoria, i)
                elif any(x in categoria for x in ["Dermocosmético", "Cremas", "Protectores"]):
                    nombre = self._generar_nombre_dermocosmetico(i)
                else:
                    nombre = self._generar_nombre_producto(categoria, i)
                
                # Determinar precio y costo
                if "Genérico" in categoria:
                    precio_base = np.random.uniform(2.5, 15.0)
                elif "OTC" in categoria or "Suplementos" in categoria:
                    precio_base = np.random.uniform(4.0, 35.0)
                elif any(x in categoria for x in ["Equipo", "Material"]):
                    precio_base = np.random.uniform(8.0, 200.0)
                elif any(x in categoria for x in ["Dermocosmético", "Cremas"]):
                    precio_base = np.random.uniform(12.0, 80.0)
                else:  # Éticos
                    precio_base = np.random.uniform(10.0, 120.0)
                
                costo = precio_base * np.random.uniform(0.55, 0.7)
                
                requiere_receta = 1 if any(x in categoria for x in [
                    "Antibióticos", "Antihipertensivos", "Antidiabéticos", "Antiasmáticos",
                    "Antidepresivos", "Antipsicóticos", "Antiepilépticos", "Antineoplásicos"
                ]) else 0
                
                productos.append({
                    'sku_id': sku_id,
                    'nombre_producto': nombre,
                    'categoria_farmaceutica': categoria,
                    'principio_activo': self._extraer_principio_activo(nombre),
                    'presentacion': self._determinar_presentacion(categoria),
                    'laboratorio': np.random.choice(labs),
                    'precio_unitario': round(precio_base, 2),
                    'costo_unitario': round(costo, 2),
                    'requiere_receta': requiere_receta,
                    'clasificacion_atc': self._obtener_codigo_atc(categoria),
                    'es_controlado': 1 if "Psic" in categoria or "Narc" in categoria else 0,
                    'temperatura_controlada': 1 if "Insulina" in nombre or "Vacuna" in nombre else 0,
                    'volumen_unidades': np.random.choice([1, 2, 3, 5, 10, 20, 30, 50, 100], 
                                                        p=[0.3, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.03, 0.02])
                })
                
                sku_counter += 1
        
        df_productos = pd.DataFrame(productos)
        
        # Verificación final
        print(f"\n✅ Catálogo generado: {len(df_productos)} SKUs")
        print(f"   • Rango SKUs: {df_productos['sku_id'].min()} - {df_productos['sku_id'].max()}")
        print(f"   • Categorías únicas: {df_productos['categoria_farmaceutica'].nunique()}")
        
        # Verificar que coincida con el número esperado
        categorias_esperadas = len(distribucion_skus)
        categorias_reales = df_productos['categoria_farmaceutica'].nunique()
        
        if categorias_reales != categorias_esperadas:
            print(f"   ⚠️ Advertencia: Categorías esperadas: {categorias_esperadas}, reales: {categorias_reales}")
        
        return df_productos

    def _generar_nombre_producto(self, categoria, index):
        # Genera nombre realista de producto farmacéutico
        if "Antibiótico" in categoria:
            principios = ["Amoxicilina", "Azitromicina", "Ciprofloxacina", "Claritromicina", "Cefalexina"]
            principio = principios[index % len(principios)]
            dosis = [250, 500, 750, 1000][(index + len(principio)) % 4]
            return f"{principio} {dosis}mg"
        
        elif "Antihipertensivo" in categoria:
            principios = ["Losartán", "Enalapril", "Amlodipina", "Valsartán", "Metoprolol"]
            principio = principios[index % len(principios)]
            dosis = [50, 100, 150][(index + len(principio)) % 3]
            return f"{principio} {dosis}mg"
        
        elif "Analgésico" in categoria:
            principios = ["Ibuprofeno", "Paracetamol", "Naproxeno", "Diclofenaco", "Ketorolaco"]
            principio = principios[index % len(principios)]
            dosis = [400, 500, 600, 800][(index + len(principio)) % 4]
            return f"{principio} {dosis}mg"
        
        else:
            marcas_comunes = ["X", "Z", "M", "N", "S", "A", "C", "F", "T", "B"]
            sufijos = ["", " Forte", " Plus", " Rapid", " SR", " XR", " Duo", " Max"]
            
            letra = marcas_comunes[index % len(marcas_comunes)]
            sufijo = sufijos[(index + len(letra)) % len(sufijos)]
            return f"{categoria[:15]} {letra}{index % 100}{sufijo}"
    
    def _generar_nombre_generico(self, categoria, index):
        #Genera nombre de medicamento genérico usando el índice para variar   
        if "Cardiovascular" in categoria:
            principios = ["Losartán", "Amlodipina", "Atorvastatina", "Enalapril", "Metoprolol"]
        elif "Sistema Nervioso" in categoria:
            principios = ["Fluoxetina", "Sertralina", "Clonazepam", "Tramadol", "Gabapentina"]
        elif "Antiinfeccioso" in categoria:
            principios = ["Amoxicilina", "Trimetoprima", "Ciprofloxacina", "Metronidazol", "Nitrofurantoína"]
        else:
            principios = ["Omeprazol", "Metformina", "Salbutamol", "Loratadina", "Hidroxicloroquina"]
        
        # Usar el índice para seleccionar el principio activo de manera determinística pero variada
        principio = principios[index % len(principios)]
        
        # Dosis varía según el índice también
        dosis_opciones = [10, 20, 40, 50, 100, 200, 400, 500, 850]
        dosis = dosis_opciones[(index + len(principio)) % len(dosis_opciones)]
        
        return f"{principio} {dosis}mg Genérico"

    def _generar_nombre_dermocosmetico(self, index):
        #Genera nombre de producto dermocosmético (sin parámetro categoria)
        prefijos = ["Crema", "Gel", "Loción", "Sérum", "Espuma", "Leche", "Bálsamo"]
        propiedades = ["Hidratante", "Anti-edad", "Reafirmante", "Iluminador", "Calmante", "Regenerador"]
        ingredientes = ["Ácido Hialurónico", "Vitamina C", "Retinol", "Niacinamida", "Ceramidas", "Centella"]
        
        # Usar el índice para variar, pero mantener aleatoriedad
        prefijo = prefijos[index % len(prefijos)]
        propiedad = propiedades[(index + len(prefijo)) % len(propiedades)]
        ingrediente = ingredientes[(index + len(propiedad)) % len(ingredientes)]
        
        return f"{prefijo} {propiedad} con {ingrediente}"

    def _extraer_principio_activo(self, nombre):
        #Extrae el principio activo del nombre del producto
        palabras = nombre.split()
        if len(palabras) > 0:
            return palabras[0]
        return "Principio Activo"
    
    def _determinar_presentacion(self, categoria):
        #Determina presentación según categoría
        if "Tableta" in categoria or "Cápsula" in categoria or "Genérico" in categoria:
            return np.random.choice(["Tabletas x10", "Tabletas x20", "Tabletas x30", "Cápsulas x30", "Cápsulas x60"])
        elif "Jarabe" in categoria or "Suspensión" in categoria:
            return np.random.choice(["Frasco 60ml", "Frasco 120ml", "Frasco 240ml"])
        elif "Crema" in categoria or "Gel" in categoria:
            return np.random.choice(["Tubo 30g", "Tubo 50g", "Tubo 100g", "Pote 200g"])
        elif "Inyectable" in categoria:
            return np.random.choice(["Ampolla 1ml", "Ampolla 2ml", "Vial 5ml"])
        elif "Inhalador" in categoria:
            return np.random.choice(["Inhalador 200 dosis", "Inhalador 120 dosis"])
        else:
            return np.random.choice(["Unidad", "Caja", "Paquete", "Kit"])
    
    def _obtener_codigo_atc(self, categoria):
        # Asigna código ATC según categoría
        codigos = {
            "Antibióticos": "J01", "Antihipertensivos": "C09", "Antidiabéticos": "A10",
            "Analgésicos": "N02", "Vitaminas": "A11", "Dermocosméticos": "D02",
            "Antiasmáticos": "R03", "Antidepresivos": "N06", "Antipsicóticos": "N05",
            "Anticonceptivos": "G03", "Corticosteroides": "H02", "Antiepilépticos": "N03"
        }
        for key, codigo in codigos.items():
            if key in categoria:
                return codigo
        return "V03"
    
    # ============= MÉTODOS DE GENERACIÓN DE FARMACIAS =============

    def _generar_farmacias(self, num_farmacias):
        """Genera farmacias con perfiles realistas"""
        farmacias = []
        
        # Distribución por región
        distribucion_regiones = {
            "Sierra": 0.45,   # 45%
            "Costa": 0.40,    # 40%
            "Oriente": 0.10,  # 10%
            "Insular": 0.05   # 5%
        }
        
        # Distribución por tipo
        tipos = ["Farmacia Comunitaria", "Farmacia Cadena", "Botica"]
        prob_tipos = [0.4, 0.5, 0.1]  # 50% son de cadena ahora
        
        for i in range(num_farmacias):
            region = np.random.choice(
                list(distribucion_regiones.keys()),
                p=list(distribucion_regiones.values())
            )
            ciudad = np.random.choice(self.ciudades[region])
            tipo = np.random.choice(tipos, p=prob_tipos)
            
            # Tamaño de farmacia (influye en portafolio)
            if tipo == "Farmacia Cadena":
                tamano = np.random.choice(["Grande", "Mediana", "Pequeña"], p=[0.3, 0.5, 0.2])
            elif tipo == "Farmacia Comunitaria":
                tamano = np.random.choice(["Mediana", "Pequeña"], p=[0.4, 0.6])
            else:  # Botica
                tamano = "Pequeña"
            
            farmacias.append({
                'farmacia_id': f"FARM{i+1:04d}",  # FARM0001 formato
                'region': region,
                'ciudad': ciudad,
                'tipo_farmacia': tipo,
                'tamano': tamano,
                'tamano_m2': self._asignar_metros_cuadrados(tipo, tamano)
            })
        
        return farmacias

    def _asignar_metros_cuadrados(self, tipo, tamano):
        """Asigna metros cuadrados según tipo y tamaño"""
        if tipo == "Farmacia Cadena":
            if tamano == "Grande":
                return np.random.randint(200, 400)
            elif tamano == "Mediana":
                return np.random.randint(120, 200)
            else:
                return np.random.randint(80, 120)
        elif tipo == "Farmacia Comunitaria":
            if tamano == "Mediana":
                return np.random.randint(80, 150)
            else:
                return np.random.randint(40, 80)
        else:  # Botica
            return np.random.randint(20, 50)
   
    def _asignar_portafolios(self, farmacias):
        """Asigna portafolios de SKUs a cada farmacia según su perfil"""
        portafolios = {}
        
        # Configuración de portafolios por tipo
        config_portafolio = {
            "Farmacia Cadena": {
                "Grande": {"min": 4200, "max": 4800},   # 4,500 promedio
                "Mediana": {"min": 3800, "max": 4200},  # 4,000 promedio
                "Pequeña": {"min": 3200, "max": 3800}   # 3,500 promedio
            },
            "Farmacia Comunitaria": {
                "Mediana": {"min": 2800, "max": 3500},  # 3,200 promedio
                "Pequeña": {"min": 2000, "max": 2800}   # 2,400 promedio
            },
            "Botica": {
                "Pequeña": {"min": 1500, "max": 2500}   # 2,000 promedio
            }
        }
        
        # Factores regionales para ajustar portafolio
        factores_regionales = {
            "Sierra": {
                "Antigripales": 1.3,
                "Antiasmáticos": 1.2,
                "Vitaminas": 1.1
            },
            "Costa": {
                "Protectores Solares": 1.4,
                "Repelentes": 1.5,
                "Antifúngicos": 1.3
            },
            "Oriente": {
                "Repelentes": 1.8,
                "Antibióticos": 1.4,
                "Antiparasitarios": 1.6
            },
            "Insular": {
                "Protectores Solares": 1.6,
                "Repelentes": 1.5,
                "Antihistamínicos": 1.3
            }
        }
        
        for farmacia in farmacias:
            config = config_portafolio[farmacia['tipo_farmacia']][farmacia['tamano']]
            num_skus = np.random.randint(config['min'], config['max'] + 1)
            
            # Seleccionar SKUs con probabilidad según categoría
            skus_seleccionados = []
            pesos = []
            
            for _, producto in self.productos_catalogo.iterrows():
                peso = 1.0
                
                # Ajuste por categoría según región
                for key, factor in factores_regionales[farmacia['region']].items():
                    if key in producto['categoria_farmaceutica']:
                        peso *= factor
                
                # Ajuste por tipo de farmacia
                if farmacia['tipo_farmacia'] == "Farmacia Cadena":
                    if "Dermocosmético" in producto['categoria_farmaceutica']:
                        peso *= 1.3
                    if "Equipo" in producto['categoria_farmaceutica']:
                        peso *= 1.2
                
                # Productos de alta rotación tienen más probabilidad
                if any(x in producto['categoria_farmaceutica'] for x in ["Analgésicos", "Antigripales", "Vitaminas"]):
                    peso *= 1.5
                
                # Genéricos tienen alta probabilidad
                if "Genérico" in producto['categoria_farmaceutica']:
                    peso *= 1.4
                
                skus_seleccionados.append(producto['sku_id'])
                pesos.append(peso)
            
            pesos = np.array(pesos) / sum(pesos)
            skus_elegidos = np.random.choice(
                skus_seleccionados,
                size=min(num_skus, len(skus_seleccionados)),
                replace=False,
                p=pesos
            )
            
            portafolio_df = self.productos_catalogo[
                self.productos_catalogo['sku_id'].isin(skus_elegidos)
            ].copy()
            
            portafolios[farmacia['farmacia_id']] = portafolio_df
            
            if len(portafolios) % 100 == 0:
                print(f"   Portafolios asignados: {len(portafolios)}/{len(farmacias)}")
        
        return portafolios

    # ============= MÉTODOS DE CÁLCULO DE DEMANDA =============

    def _get_region_factor(self, categoria, region, ciudad):
        """Factores específicos por región y ciudad"""
        region_factors = {
            "Sierra": {
                "Antigripales y Antitusivos": 1.4,
                "Vitaminas y Suplementos Alimenticios": 1.2,
                "Protectores Solares y Post-solares": 0.8,
                "Repelentes de Insectos": 0.9,
                "Monitores de Salud": 1.3
            },
            "Costa": {
                "Protectores Solares y Post-solares": 1.8,
                "Repelentes de Insectos": 1.6,
                "Antigripales y Antitusivos": 1.1,
                "Dermocosméticos Faciales": 1.4,
                "Vitaminas y Suplementos Alimenticios": 1.1
            },
            "Oriente": {
                "Repelentes de Insectos": 2.0,
                "Antibióticos Sistémicos": 1.5,
                "Material de Curación": 1.3,
                "Protectores Solares y Post-solares": 1.2
            },
            "Insular": {
                "Protectores Solares y Post-solares": 2.2,
                "Repelentes de Insectos": 1.9,
                "Antigripales y Antitusivos": 1.0,
                "Material de Curación": 1.2
            }
        }
        
        ciudad_factors = {
            "Quito": {"Monitores de Salud": 1.5, "Genéricos": 1.3},
            "Guayaquil": {"Dermocosméticos": 1.6, "OTC": 1.4},
            "Cuenca": {"Productos Naturales": 1.4, "Homeopatía": 1.5},
            "Manta": {"Protectores Solares": 2.0, "Repelentes": 1.8},
            "Puyo": {"Repelentes": 2.2, "Antibióticos": 1.6}
        }
        
        factor = 1.0
        
        if region in region_factors:
            for cat_pattern, region_factor in region_factors[region].items():
                if cat_pattern in categoria:
                    factor *= region_factor
        
        if ciudad in ciudad_factors:
            for cat_pattern, city_factor in ciudad_factors[ciudad].items():
                if cat_pattern in categoria:
                    factor *= city_factor
        
        return max(0.5, min(factor, 3.0))  # Limitar entre 0.5 y 3.0

    def generate_demand(self, categoria, fecha, region="Sierra", ciudad="Quito"):
        """Genera demanda realista para categoría farmacéutica"""
        
        # 1. Obtener distribución base
        dist_type, *params = EcuadorSeasonalityAdjusted.get_demand_distribution(categoria)
        
        # 2. Generar según distribución
        if dist_type == "poisson":
            lambda_base = params[0]
            base_demand = np.random.poisson(lambda_base)
        elif dist_type == "normal":
            mu, sigma = params[0], params[1]
            base_demand = max(0, int(np.random.normal(mu, sigma)))
        elif dist_type == "negative_binomial":
            r, p = params[0], params[1]
            base_demand = np.random.negative_binomial(r, p)
        elif dist_type == "lognormal":
            mu, sigma = params[0], params[1]
            base_demand = int(np.random.lognormal(mu, sigma))
        else:
            base_demand = np.random.poisson(20)  # Default
        
        # 3. Aplicar estacionalidad
        month = fecha.month
        seasonality_factor = EcuadorSeasonalityAdjusted.get_category_factors(categoria, month)
        
        # 4. Factores regionales
        region_factor = self._get_region_factor(categoria, region, ciudad)
        
        # 5. Factores temporales
        weekday_factor = 1.3 if fecha.weekday() >= 4 else 1.0  # +30% fines de semana
        payday_factor = 1.25 if fecha.day in [15, 30, 31] or (10 <= fecha.day <= 12) else 1.0
        
        # 6. Calcular demanda final
        final_demand = base_demand * seasonality_factor * region_factor * weekday_factor * payday_factor
        
        # 7. Redondear y limitar
        final_demand = int(np.round(final_demand))
        final_demand = max(0, min(final_demand, 100))  # Máximo 100 unidades
        
        return final_demand
    
    def _get_season(self, month):
        """Determina temporada en Ecuador"""
        if month in [12, 1, 2]:
            return "Verano"
        elif month in [3, 4, 5]:
            return "Otoño"
        elif month in [6, 7, 8]:
            return "Invierno"
        else:
            return "Primavera"

    def _get_region_factor_base(self, region, ciudad):
        # Pre-calcula factor regional base (constante para toda la simulación)
        # Implementación simplificada - retorna un factor base
        factores_regionales = {
            "Sierra": 1.1, "Costa": 1.2, "Oriente": 1.3, "Insular": 1.4
        }
        return factores_regionales.get(region, 1.0)
    
    def _generate_demand_fast(self, categoria):
        # Versión rápida de generate_demand sin validaciones extras
        dist_type, *params = EcuadorSeasonalityAdjusted.get_demand_distribution(categoria)
        
        if dist_type == "poisson":
            return np.random.poisson(params[0])
        elif dist_type == "normal":
            return max(0, int(np.random.normal(params[0], params[1])))
        elif dist_type == "negative_binomial":
            return np.random.negative_binomial(params[0], params[1])
        elif dist_type == "lognormal":
            return int(np.random.lognormal(params[0], params[1]))
        else:
            return np.random.poisson(20)

    # ============= MÉTODO PRINCIPAL DE SIMULACIÓN =============
    def simulate_dataset(self, num_farmacias=1400, num_weeks=156, 
                         start_date=datetime(2023, 1, 1),
                         output_path='data/simulacion_farmaceutica_ecuador.parquet'):
        """
        Versión BIG DATA - NO CARGA TODO EN MEMORIA y PUEDE REANUDAR DE HABER ALGUN FALLO
        Genera directamente el archivo CSV final sin crear df_final en RAM
        """
        
        print("💊 SIMULACIÓN FARMACÉUTICA ECUATORIANA - VERSIÓN BIG DATA")
        print("=" * 70)
        print(f"   • Output path: {output_path}")
        
        # Directorios
        BATCH_DIR = 'data/batches'
        OUTPUT_DIR = 'data'
        POWERBI_DIR = 'dashboard/data_farmacias'
        # Crear directorios
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(BATCH_DIR, exist_ok=True)
        os.makedirs(POWERBI_DIR, exist_ok=True)
        
        # ===== VERIFICAR ESTADO DE CONSOLIDACIÓN =====
        consolidated_files = sorted(glob.glob(os.path.join(BATCH_DIR, 'consolidated_*.parquet')))
        batch_files = sorted(glob.glob(os.path.join(BATCH_DIR, 'batch_*.parquet')))

         # CASO 1: Si hay consolidated files, convertirlos directamente
        if consolidated_files and not batch_files:
            print(f"📦 Encontrados {len(consolidated_files)} archivos consolidados")
            print("   Convirtiendo a Parquet final...")
            return self._consolidated_to_parquet(consolidated_files, output_path, BATCH_DIR)

        # CASO 2: Si hay batches pendientes Y existe el archivo final
        if batch_files and os.path.exists(output_path):
            print(f"\n🔄 MODO REANUDACIÓN DETECTADO")
            print(f"   • Archivo final existente: {output_path}")
            print(f"   • Batches pendientes: {len(batch_files)}")
            print(f"   • Rango batches: {batch_files[0]} - {batch_files[-1]}")
            
            # Verificar integridad del archivo existente
            try:
                # Solo leemos el schema, no los datos
                import pyarrow.parquet as pq
                schema = pq.read_schema(output_path)
                print(f"   • Schema del archivo existente verificado")
                print(f"   • Continuando con la consolidación...")
            except Exception as e:
                print(f"   ⚠️ Error al leer archivo existente: {e}")
                print(f"   Se procesarán todos los batches desde cero")
                # Backup del archivo corrupto
                if os.path.exists(output_path):
                    os.rename(output_path, output_path + ".corrupto")
            
            # Proceder a consolidar los batches restantes
            return self._batches_to_parquet_reanudable(batch_files, output_path, BATCH_DIR)
             
        # CASO 3: Si hay batches pendientes pero NO existe archivo final
        if batch_files and not os.path.exists(output_path):
            print(f"📦 Encontrados {len(batch_files)} batches pendientes")
            print("   Iniciando proceso de consolidación desde cero...")
            return self._batches_to_parquet_reanudable(batch_files, output_path, BATCH_DIR)
        
        # CASO 4: Si hay batches pendientes y consolidated files (raro)
        if batch_files and consolidated_files:
            print(f"⚠️ Estado mixto: {len(batch_files)} batches y {len(consolidated_files)} consolidated")
            print("   Procesando consolidated files primero...")
            self._consolidated_to_parquet(consolidated_files, output_path, BATCH_DIR)
            # Recursión para procesar batches restantes
            return self.simulate_dataset(num_farmacias, num_weeks, start_date, output_path)
    
        # ===== SI NO HAY DATOS PREVIOS, INICIAR SIMULACIÓN COMPLETA =====
        print("\n🆕 No se encontraron datos previos. Iniciando simulación completa...")

        # Generar farmacias
        farmacias = self._generar_farmacias(num_farmacias)
        
        # Asignar portafolios
        print("\n🏪 Asignando portafolios a farmacias...")
        portafolios_farmacias = self._asignar_portafolios(farmacias)
        
        # Calcular estimación
        avg_skus_por_farmacia = np.mean([len(p) for p in portafolios_farmacias.values()])
        registros_estimados = num_farmacias * num_weeks * avg_skus_por_farmacia
        print(f"\n⚙️ Simulando {registros_estimados:,.0f} registros estimados...")
        print(f"   • Farmacias: {num_farmacias}")
        print(f"   • Semanas: {num_weeks}")
        print(f"   • SKUs promedio por farmacia: {avg_skus_por_farmacia:.0f}")
        
        # ===== PARÁMETROS OPTIMIZADOS =====
        BATCH_SIZE = 500000  # 500k registros por batch
        BATCH_PRODUCTOS = 100  # Procesar 100 productos a la vez con numpy
        
        batch_counter = 0
        all_batch_files = []
        
        # Variables para seguimiento
        total_registros = 0
        start_time = datetime.now()
        
        # Datos acumulados para el batch actual
        current_batch_data = []
        
        for idx_farmacia, farmacia in enumerate(farmacias):
            farmacia_start = datetime.now()
            portafolio = portafolios_farmacias[farmacia['farmacia_id']]
            
            # Convertir portafolio a lista de diccionarios para acceso rápido
            productos_list = portafolio.to_dict('records')
            num_productos = len(productos_list)
            
            # Pre-calcular factor regional base (constante para esta farmacia)
            region_factor_base = self._get_region_factor_base(farmacia['region'], farmacia['ciudad'])
            
            # Procesar en lotes de productos para usar numpy
            for producto_idx in range(0, num_productos, BATCH_PRODUCTOS):
                batch_productos = productos_list[producto_idx:producto_idx + BATCH_PRODUCTOS]
                tam_batch = len(batch_productos)
                
                # Crear arrays para este lote de productos
                categorias_batch = [p['categoria_farmaceutica'] for p in batch_productos]
                es_generico = np.array(['Genérico' in cat for cat in categorias_batch])
                es_equipo = np.array(['Equipo' in cat for cat in categorias_batch])
                es_otc = np.array(['OTC' in cat for cat in categorias_batch])
                
                for semana in range(num_weeks):
                    fecha = start_date + timedelta(weeks=semana)
                    month = fecha.month
                    
                    # Factores semanales (vectorizado)
                    seasonality_factors = np.array([
                        EcuadorSeasonalityAdjusted.get_category_factors(cat, month)
                        for cat in categorias_batch
                    ])
                    
                    weekday_factor = 1.3 if fecha.weekday() >= 4 else 1.0
                    payday_factor = 1.25 if fecha.day in [15, 30, 31] else 1.0
                    
                    # Promociones (vectorizado)
                    prob_promo_base = 0.25
                    if month in [1, 5, 10, 11, 12]:
                        prob_promo_base *= 1.5
                    
                    # Ajuste por categoría (dermocosméticos tienen más promociones)
                    prob_promo = np.full(tam_batch, prob_promo_base)
                    es_dermocosmetico = np.array(['Dermocosmético' in cat for cat in categorias_batch])
                    prob_promo[es_dermocosmetico] *= 1.3
                    
                    promos = np.random.random(tam_batch) < prob_promo
                    
                    # Demanda base (vectorizado)
                    demandas_base = np.array([
                        self._generate_demand_fast(cat)
                        for cat in categorias_batch
                    ])
                    
                    # Aplicar factores (operaciones vectorizadas)
                    demandas = demandas_base.astype(float)
                    demandas *= seasonality_factors
                    demandas *= region_factor_base
                    demandas *= weekday_factor
                    demandas *= payday_factor
                    
                    # Ajustes por tipo de producto (vectorizado con máscaras)
                    if np.any(es_generico):
                        demandas[es_generico] *= np.random.uniform(0.8, 1.2, np.sum(es_generico))
                    
                    if np.any(es_equipo):
                        demandas[es_equipo] = np.maximum(1, demandas[es_equipo] * 0.3)
                    
                    if np.any(promos):
                        demandas[promos] *= np.random.uniform(1.2, 1.6, np.sum(promos))
                    
                    # Limitar y redondear
                    demandas = np.minimum(np.round(demandas).astype(int), 50)
                    
                    # Errores de predicción (vectorizado)
                    error_levels = np.where(es_otc, 0.15, 0.25)
                    
                    errores_arima = np.random.normal(0, demandas * error_levels * 1.2)
                    errores_prophet = np.random.normal(0, demandas * error_levels * 1.0)
                    errores_lgb = np.random.normal(0, demandas * error_levels * 0.8)
                    
                    # Construir registros para este lote/semana
                    for j, producto in enumerate(batch_productos):
                        current_batch_data.append({
                            'fecha': fecha.strftime('%Y-%m-%d'),  # Guardar como string para CSV
                            'farmacia_id': farmacia['farmacia_id'],
                            'sku_id': producto['sku_id'],
                            'nombre_producto': producto['nombre_producto'],
                            'region': farmacia['region'],
                            'ciudad': farmacia['ciudad'],
                            'tipo_farmacia': farmacia['tipo_farmacia'],
                            'tamano_farmacia': farmacia['tamano'],
                            'categoria_farmaceutica': producto['categoria_farmaceutica'],
                            'principio_activo': producto['principio_activo'],
                            'presentacion': producto['presentacion'],
                            'laboratorio': producto['laboratorio'],
                            'precio_unitario': producto['precio_unitario'],
                            'costo_unitario': producto['costo_unitario'],
                            'requiere_receta': producto['requiere_receta'],
                            'clasificacion_atc': producto['clasificacion_atc'],
                            'promocion': int(promos[j]),
                            'demanda_real': int(demandas[j]),
                            'pred_arima': max(0, int(demandas[j] + errores_arima[j])),
                            'pred_prophet': max(0, int(demandas[j] + errores_prophet[j])),
                            'pred_lightgbm': max(0, int(demandas[j] + errores_lgb[j])),
                            'mes': month,
                            'semana_anio': fecha.isocalendar()[1],
                            'dia_semana': fecha.weekday(),
                            'es_fin_semana': 1 if fecha.weekday() >= 4 else 0,
                            'es_quincena': 1 if fecha.day in [15, 30, 31] else 0,
                            'temporada': self._get_season(month)
                        })
                        
                        total_registros += 1
                        
                        # ===== GUARDAR BATCH SI ALCANZAMOS EL TAMAÑO =====
                        if len(current_batch_data) >= BATCH_SIZE:
                            batch_file = f'data/batches/batch_{batch_counter:06d}.parquet'
                            pd.DataFrame(current_batch_data).to_parquet(batch_file, compression='snappy')
                            all_batch_files.append(batch_file)
                            print(f"\n   💾 Batch {batch_counter} guardado: {len(current_batch_data):,} registros")
                            current_batch_data = []
                            batch_counter += 1
                            
                            # Mostrar progreso
                            elapsed = (datetime.now() - start_time).total_seconds()
                            rate = total_registros / elapsed if elapsed > 0 else 0
                            print(f"   Progreso: {total_registros:,.0f} registros | {rate:,.0f} reg/seg | Batches: {batch_counter}")
            
            # Mostrar progreso por farmacia
            farmacia_time = (datetime.now() - farmacia_start).total_seconds()
            print(f"   ✅ Farmacia {idx_farmacia + 1}/{num_farmacias} completada en {farmacia_time:.1f}s")
        
        # ===== GUARDAR DATOS RESTANTES AL FINAL =====
        if current_batch_data:
            batch_file = f'data/batches/batch_{batch_counter:06d}.parquet'
            pd.DataFrame(current_batch_data).to_parquet(batch_file, compression='snappy')
            all_batch_files.append(batch_file)
            print(f"\n   💾 Batch final {batch_counter} guardado: {len(current_batch_data):,} registros")
            batch_counter += 1
        
        print(f"\n\n✅ Simulación base completada: {total_registros:,} registros en {batch_counter} batches")
        
        # Al final, llamar al nuevo método de consolidación
        print("\n🔄 Consolidando batches a Parquet final...")
        return self._batches_to_parquet_reanudable(all_batch_files, output_path, BATCH_DIR)

    def _batches_to_parquet_reanudable(self, batch_files, output_path, BATCH_DIR):
        """
        Convierte batches a Parquet final SIN CARGAR EN MEMORIA.
        Si el archivo ya existe, agrega nuevos row groups.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        print(f"\n📦 Consolidando {len(batch_files)} batches a Parquet (modo reanudable)...")
        
        total_registros = 0
        start_time = datetime.now()
        writer = None
        schema = None
        archivo_existente = os.path.exists(output_path)
        
        try:
            # CASO 1: El archivo ya existe - obtener su schema
            if archivo_existente:
                print(f"   📂 Archivo existente detectado: {output_path}")
                
                # Leer SOLO el schema, no los datos
                schema = pq.read_schema(output_path)
                
                # Obtener número de registros aproximado (solo metadatos)
                try:
                    parquet_file = pq.ParquetFile(output_path)
                    total_registros = parquet_file.metadata.num_rows
                    num_row_groups = parquet_file.num_row_groups
                    print(f"   • Registros existentes: {total_registros:,}")
                    print(f"   • Row groups existentes: {num_row_groups}")
                    print(f"   • Agregando {len(batch_files)} nuevos batches como row groups adicionales")
                except:
                    print(f"   ⚠️ No se pudo leer metadatos, continuando de todos modos")
                
                # Abrir en modo append (escribiendo nuevos row groups)
                writer = pq.ParquetWriter(output_path, schema, compression='snappy')
            
            # Procesar cada batch pendiente
            for i, batch_file in enumerate(batch_files):
                if not os.path.exists(batch_file):
                    print(f"\n   ⚠️ Batch no encontrado: {batch_file}")
                    continue
                
                print(f"   Procesando batch {i+1}/{len(batch_files)}: {os.path.basename(batch_file)}...", end='\r')
                
                # Cargar batch
                df_batch = pd.read_parquet(batch_file)
                batch_size = len(df_batch)
                
                # Calcular errores si es necesario
                if 'error_arima' not in df_batch.columns:
                    df_batch['error_arima'] = abs(df_batch['demanda_real'] - df_batch['pred_arima'])
                    df_batch['error_prophet'] = abs(df_batch['demanda_real'] - df_batch['pred_prophet'])
                    df_batch['error_lightgbm'] = abs(df_batch['demanda_real'] - df_batch['pred_lightgbm'])
                
                # Asegurar formato fecha
                if 'fecha' in df_batch.columns and not isinstance(df_batch['fecha'].iloc[0], str):
                    df_batch['fecha'] = df_batch['fecha'].astype(str)
                
                # Convertir a tabla Arrow
                table = pa.Table.from_pandas(df_batch)
                
                # CASO 2: Primer batch y NO existe archivo
                if writer is None and not archivo_existente:
                    schema = table.schema
                    writer = pq.ParquetWriter(output_path, schema, compression='snappy')
                    writer.write_table(table)
                    print(f"\n   ✅ Primer batch escrito: {batch_size:,} registros")
                else:
                    # Verificar schema
                    if table.schema == schema:
                        writer.write_table(table)
                    else:
                        print(f"\n   ⚠️ Schema mismatch. Forzando conversión...")
                        table = table.cast(schema)
                        writer.write_table(table)
                
                total_registros += batch_size
                
                # Liberar memoria
                del df_batch
                del table
                
                # Eliminar batch procesado SOLO si se escribió correctamente
                try:
                    os.remove(batch_file)
                    # Crear archivo de control para saber hasta dónde vamos
                    control_file = os.path.join(BATCH_DIR, f"ultimo_batch_procesado.txt")
                    with open(control_file, 'w') as f:
                        f.write(f"{batch_file}\n")
                        f.write(f"registros_totales={total_registros}\n")
                        f.write(f"ultima_actualizacion={datetime.now().isoformat()}")
                except Exception as e:
                    print(f"\n   ⚠️ No se pudo eliminar {batch_file}: {e}")
                
                # Mostrar progreso
                if (i + 1) % 10 == 0 or (i + 1) == len(batch_files):
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = total_registros / elapsed if elapsed > 0 else 0
                    
                    if os.path.exists(output_path):
                        size_mb = os.path.getsize(output_path) / (1024 * 1024)
                        size_gb = size_mb / 1024
                        
                        # Calcular registros totales (existentes + nuevos)
                        registros_totales = total_registros
                        if archivo_existente:
                            try:
                                pf = pq.ParquetFile(output_path)
                                registros_totales = pf.metadata.num_rows
                            except:
                                pass
                        
                        print(f"\n   ✅ {i+1}/{len(batch_files)} batches | "
                            f"Total: {registros_totales:,} reg | "
                            f"{size_mb:.1f} MB ({size_gb:.2f} GB) | "
                            f"{rate:,.0f} reg/seg")
        
        except Exception as e:
            print(f"\n❌ Error durante la consolidación: {e}")
            print("   El archivo Parquet puede estar incompleto pero legible.")
            raise
        
        finally:
            if writer:
                writer.close()
                print(f"\n   ✅ Writer cerrado correctamente")
        
        # Verificación final
        if os.path.exists(output_path):
            final_size = os.path.getsize(output_path) / (1024**3)
            
            # Obtener conteo final preciso
            try:
                pf = pq.ParquetFile(output_path)
                registros_finales = pf.metadata.num_rows
                row_groups = pf.num_row_groups
            except:
                registros_finales = total_registros
                row_groups = "desconocido"
            
            print(f"\n\n✅ CONSOLIDACIÓN COMPLETADA")
            print(f"   • Archivo final: {output_path}")
            print(f"   • Registros totales: {registros_finales:,}")
            print(f"   • Row groups: {row_groups}")
            print(f"   • Tamaño: {final_size:.2f} GB")
            print(f"   • Tiempo total: {(datetime.now() - start_time).total_seconds():.1f} segundos")
            
            # Eliminar archivo de control
            control_file = os.path.join(BATCH_DIR, "ultimo_batch_procesado.txt")
            if os.path.exists(control_file):
                os.remove(control_file)
        else:
            print(f"\n❌ Error: No se generó el archivo {output_path}")
        
        return output_path
    
    def _batches_to_parquet(self, batch_files, output_path, BATCH_DIR):
        # Convierte batches directamente a Parquet (versión simple)
        
        print(f"\n📦 Convirtiendo {len(batch_files)} batches a Parquet...")
        
        first_batch = True
        total_registros = 0
        start_time = datetime.now()
        
        for i, batch_file in enumerate(batch_files):
            print(f"   Procesando batch {i+1}/{len(batch_files)}...", end='\r')
            
            if not os.path.exists(batch_file):
                continue
            
            # Cargar batch
            df_batch = pd.read_parquet(batch_file)
            total_registros += len(df_batch)
            
            # Calcular errores
            if 'error_arima' not in df_batch.columns:
                df_batch['error_arima'] = abs(df_batch['demanda_real'] - df_batch['pred_arima'])
                df_batch['error_prophet'] = abs(df_batch['demanda_real'] - df_batch['pred_prophet'])
                df_batch['error_lightgbm'] = abs(df_batch['demanda_real'] - df_batch['pred_lightgbm'])
            
            # Asegurar formato fecha como string
            if 'fecha' in df_batch.columns and not isinstance(df_batch['fecha'].iloc[0], str):
                df_batch['fecha'] = df_batch['fecha'].astype(str)
            
            # Guardar
            if first_batch:
                df_batch.to_parquet(output_path, compression='snappy', index=False)
                first_batch = False
            else:
                # Leer existente, concatenar y guardar
                df_existing = pd.read_parquet(output_path)
                df_combined = pd.concat([df_existing, df_batch], ignore_index=True)
                df_combined.to_parquet(output_path, compression='snappy', index=False)
                del df_existing
                del df_combined
            
            # Eliminar batch procesado
            try:
                os.remove(batch_file)
            except:
                pass
            
            # Progreso cada 10 batches
            if (i + 1) % 10 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = total_registros / elapsed if elapsed > 0 else 0
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"\n   ✅ {i+1}/{len(batch_files)} batches | {total_registros:,} reg | {size_mb:.1f} MB | {rate:,.0f} reg/seg")
        
        print(f"\n\n✅ Parquet final: {output_path} ({total_registros:,} registros, {os.path.getsize(output_path)/1024**3:.2f} GB)")
        return output_path

    def _batches_to_consolidated(self, batch_files, BATCH_DIR):
        """
        Convierte batches a archivos consolidated (checkpoints)
        Similar a resume_consolidation pero sin crear df_final
        """
        
        ERROR_BATCH_SIZE = 500000
        checkpoint_counter = 0
        
        # Procesar batches secuencialmente
        batch_idx = 0
        total_registros = 0
        start_time = datetime.now()
        
        # Inicializar acumulador
        df_acumulado = pd.DataFrame()
        temp_consolidated_files = []
        
        while batch_idx < len(batch_files):
            current_batch_file = batch_files[batch_idx]
            
            if not os.path.exists(current_batch_file):
                batch_idx += 1
                continue
            
            try:
                df_siguiente = pd.read_parquet(current_batch_file)
            except Exception as e:
                print(f"   ⚠️ Error leyendo {current_batch_file}: {e}")
                batch_idx += 1
                continue
            
            # Calcular errores si es necesario
            if 'error_arima' not in df_siguiente.columns:
                df_siguiente['error_arima'] = abs(df_siguiente['demanda_real'] - df_siguiente['pred_arima'])
                df_siguiente['error_prophet'] = abs(df_siguiente['demanda_real'] - df_siguiente['pred_prophet'])
                df_siguiente['error_lightgbm'] = abs(df_siguiente['demanda_real'] - df_siguiente['pred_lightgbm'])
            
            # Concatenar
            if df_acumulado.empty:
                df_acumulado = df_siguiente
            else:
                df_acumulado = pd.concat([df_acumulado, df_siguiente], ignore_index=True)
            
            total_registros += len(df_siguiente)
            del df_siguiente
            
            # Si alcanzamos tamaño de checkpoint, guardar
            if len(df_acumulado) >= ERROR_BATCH_SIZE:
                checkpoint_file = os.path.join(BATCH_DIR, f'consolidated_{checkpoint_counter:06d}.parquet')
                df_acumulado.to_parquet(checkpoint_file, compression='snappy')
                temp_consolidated_files.append(checkpoint_file)
                
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = total_registros / elapsed if elapsed > 0 else 0
                print(f"   💾 Checkpoint {checkpoint_counter} guardado: {len(df_acumulado):,} reg | Total: {total_registros:,} | {rate:,.0f} reg/seg")
                
                # Eliminar batches procesados
                for idx in range(batch_idx + 1):
                    f = batch_files[idx]
                    if os.path.exists(f):
                        try:
                            os.remove(f)
                        except:
                            pass
                
                # Actualizar lista
                batch_files = [f for f in batch_files[batch_idx+1:] if os.path.exists(f)]
                batch_idx = 0
                
                # Reiniciar acumulador
                del df_acumulado
                df_acumulado = pd.DataFrame()
                checkpoint_counter += 1
            else:
                batch_idx += 1
        
        # Guardar último fragmento
        if not df_acumulado.empty:
            checkpoint_file = os.path.join(BATCH_DIR, f'consolidated_{checkpoint_counter:06d}.parquet')
            df_acumulado.to_parquet(checkpoint_file, compression='snappy')
            temp_consolidated_files.append(checkpoint_file)
            print(f"\n   💾 Checkpoint final guardado: {len(df_acumulado):,} registros")
            
            # Eliminar batches restantes
            for f in batch_files:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except:
                        pass
        
        return temp_consolidated_files

    def _consolidated_to_parquet(self, consolidated_files, output_path, BATCH_DIR):
        # Versión ultra rápida - Escribe cada archivo como row group
        
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        print(f"\n📦 Convirtiendo {len(consolidated_files)} consolidated files a Parquet...")
        
        # Configurar escritor de Parquet
        first_file = True
        total_registros = 0
        start_time = datetime.now()
        writer = None
        
        try:
            for i, cf in enumerate(consolidated_files):
                print(f"   Procesando {i+1}/{len(consolidated_files)}: {os.path.basename(cf)}...", end='\r')
                
                if not os.path.exists(cf):
                    continue
                
                # Cargar consolidated
                df_temp = pd.read_parquet(cf)
                total_registros += len(df_temp)
                
                # Calcular errores
                if 'error_arima' not in df_temp.columns:
                    df_temp['error_arima'] = abs(df_temp['demanda_real'] - df_temp['pred_arima'])
                    df_temp['error_prophet'] = abs(df_temp['demanda_real'] - df_temp['pred_prophet'])
                    df_temp['error_lightgbm'] = abs(df_temp['demanda_real'] - df_temp['pred_lightgbm'])
                
                # Asegurar formato fecha
                if 'fecha' in df_temp.columns and not isinstance(df_temp['fecha'].iloc[0], str):
                    df_temp['fecha'] = df_temp['fecha'].astype(str)
                
                # Convertir a tabla Arrow
                table = pa.Table.from_pandas(df_temp)
                
                if first_file:
                    # Primer archivo: crear writer y escribir primer row group
                    writer = pq.ParquetWriter(output_path, table.schema, compression='snappy')
                    writer.write_table(table)
                    first_file = False
                else:
                    # Escribir como nuevo row group
                    writer.write_table(table)
                
                # Eliminar archivo procesado
                try:
                    os.remove(cf)
                except:
                    pass
                
                # Liberar memoria
                del df_temp
                del table
                
                # Progreso
                if (i + 1) % 10 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = total_registros / elapsed if elapsed > 0 else 0
                    size_mb = os.path.getsize(output_path) / (1024 * 1024) if os.path.exists(output_path) else 0
                    print(f"\n   ✅ {i+1}/{len(consolidated_files)} | {total_registros:,} reg | {size_mb:.1f} MB | {rate:,.0f} reg/seg")
        
        finally:
            if writer:
                writer.close()
        
        print(f"\n\n✅ Parquet final: {output_path} ({total_registros:,} registros, {os.path.getsize(output_path)/1024**3:.2f} GB)")
        return output_path


def compress_csv_to_parquet_after_completion(csv_path, parquet_path, chunk_size=500000):
    """
    Convierte CSV gigante a Parquet comprimido después de generarlo
    """
    print(f"\n🗜️ Comprimiendo {csv_path} a Parquet...")
    
    # Verificar espacio
    csv_size = os.path.getsize(csv_path) / (1024**3)
    print(f"   • CSV actual: {csv_size:.2f} GB")
    
    # Estimar espacio necesario para Parquet
    estimated_parquet = csv_size * 0.3  # Parquet suele ser 70% más pequeño
    print(f"   • Parquet estimado: {estimated_parquet:.2f} GB")
    
    free_space = psutil.disk_usage('/').free / (1024**3)
    print(f"   • Espacio libre: {free_space:.2f} GB")
    
    if free_space < estimated_parquet * 1.5:
        print("   ⚠️ Espacio insuficiente para comprimir ahora")
        return csv_path
    
    # Convertir por chunks
    writer = None
    total_rows = 0
    
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        table = pa.Table.from_pandas(chunk)
        
        if writer is None:
            writer = pq.ParquetWriter(parquet_path, table.schema, 
                                     compression='snappy', 
                                     use_dictionary=True)
        writer.write_table(table)
        total_rows += len(chunk)
        print(f"   • Procesados {total_rows:,} registros...", end='\r')
    
    if writer:
        writer.close()
    
    # Verificar compresión
    parquet_size = os.path.getsize(parquet_path) / (1024**3)
    ratio = parquet_size / csv_size
    
    print(f"\n   ✅ Compresión completada:")
    print(f"      • Parquet: {parquet_size:.2f} GB")
    print(f"      • Ratio compresión: {ratio:.2%}")
    print(f"      • Ahorro: {(1-ratio)*100:.1f}%")
    
    # Preguntar si eliminar CSV original
    respuesta = input(f"\n¿Eliminar CSV original ({csv_size:.2f} GB) para liberar espacio? (s/N): ")
    if respuesta.lower() == 's':
        os.remove(csv_path)
        print(f"   ✅ CSV original eliminado")
    
    return parquet_path

def prepare_for_powerbi_bigdata(parquet_path, output_dir='dashboard/data_farmacias', chunk_size=1000000):
    """
    Prepara datos para Power BI procesando el Parquet por chunks
    sin cargar todo en memoria.
    
    Args:
        parquet_path: Ruta al archivo Parquet gigante
        output_dir: Directorio de salida para los CSVs
        chunk_size: Tamaño de cada chunk para procesamiento
    """
    import pyarrow.parquet as pq
    from collections import defaultdict
    
    print(f"\n🔧 PREPARANDO DATOS PARA POWER BI (MODO BIG DATA)")
    print("=" * 70)
    print(f"   • Archivo origen: {parquet_path}")
    print(f"   • Chunk size: {chunk_size:,} registros")
    print(f"   • Directorio salida: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ===== 1. EXTRAER DIMENSIONES (usando solo los primeros chunks) =====
    print("\n📊 Extrayendo tablas de dimensiones...")
    
    # Conjuntos para datos únicos
    productos_set = {}
    tiendas_set = {}
    fechas_set = set()
    skus_unicos = set()
    farmacias_unicas = set()
    
    # Leer solo los primeros chunks para dimensiones
    pf = pq.ParquetFile(parquet_path)
    total_row_groups = pf.num_row_groups
    
    print(f"   • Total row groups: {total_row_groups}")
    
    # Procesar suficientes chunks para capturar todas las dimensiones
    # (asumimos que con 10 chunks o 10M registros es suficiente)
    max_chunks_dim = min(10, total_row_groups)
    
    for i in range(max_chunks_dim):
        print(f"   • Leyendo chunk {i+1}/{max_chunks_dim} para dimensiones...", end='\r')
        table = pf.read_row_group(i)
        df_chunk = table.to_pandas()
        
        # Acumular productos únicos
        for _, row in df_chunk.iterrows():
            sku = row['sku_id']
            if sku not in productos_set:
                productos_set[sku] = {
                    'sku_id': sku,
                    'nombre_producto': row['nombre_producto'],
                    'categoria_farmaceutica': row['categoria_farmaceutica'],
                    'principio_activo': row['principio_activo'],
                    'presentacion': row['presentacion'],
                    'laboratorio': row['laboratorio'],
                    'precio_unitario': row['precio_unitario'],
                    'costo_unitario': row['costo_unitario'],
                    'requiere_receta': row['requiere_receta'],
                    'clasificacion_atc': row['clasificacion_atc']
                }
            
            # Acumular tiendas únicas
            farm_id = row['farmacia_id']
            if farm_id not in tiendas_set:
                tiendas_set[farm_id] = {
                    'farmacia_id': farm_id,
                    'region': row['region'],
                    'ciudad': row['ciudad'],
                    'tipo_farmacia': row['tipo_farmacia']
                }
            
            # Acumular fechas únicas
            fechas_set.add(row['fecha'])
        
        del df_chunk
        del table
    
    print(f"\n   • Productos únicos encontrados: {len(productos_set):,}")
    print(f"   • Tiendas únicas encontradas: {len(tiendas_set):,}")
    print(f"   • Fechas únicas encontradas: {len(fechas_set):,}")
    
    # ===== 2. GUARDAR DIMENSIONES =====
    
    # Productos
    df_productos = pd.DataFrame(list(productos_set.values()))
    df_productos.to_csv(f'{output_dir}/dim_productos.csv', index=False)
    print(f"   ✅ dim_productos.csv guardado ({len(df_productos):,} productos)")
    
    # Tiendas
    df_tiendas = pd.DataFrame(list(tiendas_set.values()))
    df_tiendas.to_csv(f'{output_dir}/dim_tiendas.csv', index=False)
    print(f"   ✅ dim_tiendas.csv guardado ({len(df_tiendas):,} tiendas)")
    
    # Fechas
    df_fechas_list = []
    for fecha_str in fechas_set:
        try:
            fecha = pd.to_datetime(fecha_str)
            df_fechas_list.append({
                'fecha': fecha_str,
                'anio': fecha.year,
                'mes': fecha.month,
                'semana': fecha.isocalendar()[1],
                'dia_semana': fecha.weekday(),
                'es_fin_semana': 1 if fecha.weekday() >= 5 else 0,
                'es_feriado': 0,  # Puedes personalizar después
                'trimestre': fecha.quarter,
                'nombre_mes': fecha.strftime('%B'),
                'nombre_dia': fecha.strftime('%A')
            })
        except:
            continue
    
    df_fechas = pd.DataFrame(df_fechas_list)
    df_fechas.to_csv(f'{output_dir}/dim_fechas.csv', index=False)
    print(f"   ✅ dim_fechas.csv guardado ({len(df_fechas):,} fechas)")
    
    # Modelos (estático)
    modelos = pd.DataFrame({
        'modelo_id': [1, 2, 3, 4],
        'nombre': ['SARIMA', 'Prophet', 'LightGBM', 'REAL'],
        'descripcion': ['Modelo SARIMA', 'Facebook Prophet', 'LightGBM', 'Datos reales'],
        'color_hex': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    })
    modelos.to_csv(f'{output_dir}/dim_modelos.csv', index=False)
    print(f"   ✅ dim_modelos.csv guardado ({len(modelos):,} modelos)")
    
    # ===== 3. PROCESAR FACT TABLES POR CHUNKS CON CHECKPOINTS =====
    print("\n📈 Procesando tablas de hechos (fact tables)...")
    
    # Archivos de salida
    fact_demanda_file = f'{output_dir}/fact_demanda.csv'
    fact_metricas_file = f'{output_dir}/metricas_modelos.csv'
    
    # Archivo de checkpoint para saber por dónde vamos
    checkpoint_file = f'{output_dir}/checkpoint.txt'
    
    # Verificar si ya hay un checkpoint (por si se interrumpió antes)
    ultimo_row_group_procesado = -1
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                ultimo_row_group_procesado = int(f.read().strip())
            print(f"   🔄 Checkpoint encontrado: reanudando desde row group {ultimo_row_group_procesado + 1}")
        except:
            ultimo_row_group_procesado = -1
    
    # Si no existen los archivos, escribir headers
    if not os.path.exists(fact_demanda_file):
        with open(fact_demanda_file, 'w', encoding='utf-8') as f:
            f.write('fecha,farmacia_id,sku_id,modelo,valor,error,promocion,fin_semana,quincena,temporada\n')
        print(f"   • Creando nuevo archivo: fact_demanda.csv")
    
    if not os.path.exists(fact_metricas_file):
        with open(fact_metricas_file, 'w', encoding='utf-8') as f:
            f.write('fecha_calculo,farmacia_id,sku_id,modelo,mae,rmse,mape\n')
        print(f"   • Creando nuevo archivo: metricas_modelos.csv")
    
    # Procesar por row groups
    total_registros_procesados = 0
    metricas_acumuladas = []
    start_time = datetime.now()
    
    # Si hay checkpoint, calcular registros ya procesados aproximados
    if ultimo_row_group_procesado >= 0:
        registros_por_rg = 500000  # Aproximado
        total_registros_procesados = (ultimo_row_group_procesado + 1) * registros_por_rg
    
    for rg_idx in range(total_row_groups):
        # Saltar si ya fue procesado
        if rg_idx <= ultimo_row_group_procesado:
            continue
        
        # Mostrar progreso (UNA SOLA LÍNEA, actualizada)
        print(f"   • Procesando row group {rg_idx+1}/{total_row_groups}... "
              f"({((rg_idx+1)/total_row_groups)*100:.1f}%)", end='\r')
        
        try:
            # Leer row group
            table = pf.read_row_group(rg_idx)
            df_chunk = table.to_pandas()
            
            # Preparar datos para fact_demanda (formato largo)
            fact_rows = []
            
            for _, row in df_chunk.iterrows():
                # Datos reales
                fact_rows.append({
                    'fecha': row['fecha'],
                    'farmacia_id': row['farmacia_id'],
                    'sku_id': row['sku_id'],
                    'modelo': 'REAL',
                    'valor': row['demanda_real'],
                    'error': 0,
                    'promocion': row['promocion'],
                    'fin_semana': row['es_fin_semana'],
                    'quincena': row['es_quincena'],
                    'temporada': row['temporada']
                })
                
                # Predicciones
                for modelo, col in [('SARIMA', 'pred_arima'), 
                                   ('Prophet', 'pred_prophet'), 
                                   ('LightGBM', 'pred_lightgbm')]:
                    fact_rows.append({
                        'fecha': row['fecha'],
                        'farmacia_id': row['farmacia_id'],
                        'sku_id': row['sku_id'],
                        'modelo': modelo,
                        'valor': row[col],
                        'error': abs(row['demanda_real'] - row[col]),
                        'promocion': row['promocion'],
                        'fin_semana': row['es_fin_semana'],
                        'quincena': row['es_quincena'],
                        'temporada': row['temporada']
                    })
            
            # Escribir fact_demanda (append)
            df_fact_chunk = pd.DataFrame(fact_rows)
            df_fact_chunk.to_csv(fact_demanda_file, mode='a', header=False, index=False)
            
            # Calcular métricas (solo cada 10 row groups para no ralentizar)
            if rg_idx % 10 == 0:
                for modelo in ['SARIMA', 'Prophet', 'LightGBM']:
                    for (farmacia_id, sku_id), group in df_chunk.groupby(['farmacia_id', 'sku_id']):
                        if len(group) > 5:  # Mínimo 5 puntos para métricas confiables
                            y_real = group['demanda_real'].values
                            
                            if modelo == 'SARIMA':
                                y_pred = group['pred_arima'].values
                            elif modelo == 'Prophet':
                                y_pred = group['pred_prophet'].values
                            else:
                                y_pred = group['pred_lightgbm'].values
                            
                            mae = np.mean(np.abs(y_real - y_pred))
                            rmse = np.sqrt(np.mean((y_real - y_pred) ** 2))
                            
                            with np.errstate(divide='ignore', invalid='ignore'):
                                mape = np.mean(np.abs((y_real - y_pred) / np.where(y_real == 0, 1, y_real))) * 100
                            
                            metricas_acumuladas.append({
                                'fecha_calculo': datetime.now().strftime('%Y-%m-%d'),
                                'farmacia_id': farmacia_id,
                                'sku_id': sku_id,
                                'modelo': modelo,
                                'mae': round(mae, 2),
                                'rmse': round(rmse, 2),
                                'mape': round(mape, 2)
                            })
            
            total_registros_procesados += len(df_chunk)
            
            # Escribir checkpoint después de cada row group exitoso
            with open(checkpoint_file, 'w') as f:
                f.write(str(rg_idx))
            
            # Escribir métricas cada cierto número
            if len(metricas_acumuladas) >= 50000:
                df_metricas_chunk = pd.DataFrame(metricas_acumuladas)
                df_metricas_chunk.to_csv(fact_metricas_file, mode='a', header=False, index=False)
                metricas_acumuladas = []
            
            # Liberar memoria
            del df_chunk
            del table
            del df_fact_chunk
            del fact_rows
            
            # Mostrar progreso detallado cada 10 row groups
            if (rg_idx + 1) % 10 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = total_registros_procesados / elapsed if elapsed > 0 else 0
                
                # Tamaños de archivo
                demanda_size = os.path.getsize(fact_demanda_file) / (1024 * 1024 * 1024)  # GB
                metricas_size = os.path.getsize(fact_metricas_file) / (1024 * 1024) if os.path.exists(fact_metricas_file) else 0  # MB
                
                # Línea clara de progreso (sin repeticiones)
                print(f"\n   ✅ {rg_idx+1}/{total_row_groups} row groups | "
                      f"{total_registros_procesados:,} reg procesados | "
                      f"Demanda: {demanda_size:.2f} GB | "
                      f"Métricas: {metricas_size:.1f} MB | "
                      f"{rate:,.0f} reg/seg")
        
        except Exception as e:
            print(f"\n   ❌ Error en row group {rg_idx+1}: {e}")
            print(f"   Checkpoint guardado en {rg_idx}. Puedes reanudar después.")
            # Guardar métricas pendientes antes de salir
            if metricas_acumuladas:
                df_metricas_chunk = pd.DataFrame(metricas_acumuladas)
                df_metricas_chunk.to_csv(fact_metricas_file, mode='a', header=False, index=False)
            raise
    
    # Escribir métricas restantes
    if metricas_acumuladas:
        df_metricas_chunk = pd.DataFrame(metricas_acumuladas)
        df_metricas_chunk.to_csv(fact_metricas_file, mode='a', header=False, index=False)
    
    # Eliminar checkpoint al terminar exitosamente
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    # ===== 4. RESUMEN FINAL =====
    print(f"\n\n✅ PREPARACIÓN COMPLETADA PARA POWER BI")
    print("=" * 70)
    print(f"Archivos generados en: {output_dir}")
    print(f"   • dim_productos.csv: {os.path.getsize(f'{output_dir}/dim_productos.csv')/1024**2:.1f} MB")
    print(f"   • dim_tiendas.csv: {os.path.getsize(f'{output_dir}/dim_tiendas.csv')/1024**2:.1f} MB")
    print(f"   • dim_fechas.csv: {os.path.getsize(f'{output_dir}/dim_fechas.csv')/1024**2:.1f} MB")
    print(f"   • dim_modelos.csv: {os.path.getsize(f'{output_dir}/dim_modelos.csv')/1024**2:.1f} MB")
    print(f"   • fact_demanda.csv: {os.path.getsize(fact_demanda_file)/1024**3:.2f} GB")
    print(f"   • metricas_modelos.csv: {os.path.getsize(fact_metricas_file)/1024**2:.1f} MB")
    
    print(f"\n📊 INSTRUCCIONES PARA POWER BI:")
    print("   1. En Power BI Desktop, usa 'Obtener datos' > 'CSV'")
    print("   2. Carga primero las tablas de dimensiones (dim_*)")
    print("   3. Luego carga fact_demanda.csv (puede tomar unos minutos)")
    print("   4. Crea relaciones:")
    print("      • fact_demanda[farmacia_id] → dim_tiendas[farmacia_id]")
    print("      • fact_demanda[sku_id] → dim_productos[sku_id]")
    print("      • fact_demanda[fecha] → dim_fechas[fecha]")
    print("      • fact_demanda[modelo] → dim_modelos[nombre]")
    
    # Al final de prepare_for_powerbi_bigdata, después de mostrar el resumen:
    print(f"\n🔄 Verificando espacio para compresión...")
    free_space = psutil.disk_usage('/').free / (1024**3)
    csv_path = f'{output_dir}/fact_demanda.csv'

    if os.path.exists(csv_path):
        csv_size = os.path.getsize(csv_path) / (1024**3)
        parquet_path = f'{output_dir}/fact_demanda.parquet'
        
        print(f"   • CSV actual: {csv_size:.2f} GB")
        print(f"   • Espacio libre: {free_space:.2f} GB")
        
        # Estimar Parquet (30% del CSV)
        estimated_parquet = csv_size * 0.3
        
        if free_space > estimated_parquet * 1.5:
            print(f"\n🗜️  Comprimiendo automáticamente a Parquet...")
            compress_csv_to_parquet_after_completion(csv_path, parquet_path)
        else:
            print(f"\n⚠️  Espacio insuficiente para comprimir ahora")
            print(f"   Necesitas {estimated_parquet * 1.5:.2f} GB, tienes {free_space:.2f} GB")
            print(f"   Puedes comprimir manualmente después con: python comprimir_fact_demanda.py")

def main():
    """Ejecutar simulación farmacéutica ecuatoriana"""
    
    print("🚀 INICIANDO SIMULACIÓN FARMACÉUTICA ECUATORIANA")
    print("=" * 70)
    
    simulator = EcuadorPharmaSimulator(seed=42)
    
    # Parámetros
    output_path = 'data/simulacion_farmaceutica_ecuador.parquet'
    powerbi_dir = 'dashboard/data_farmacias'
    
    # Crear directorios
    os.makedirs('data', exist_ok=True)
    os.makedirs(powerbi_dir, exist_ok=True)
    
    # Verificar si ya existe el archivo Parquet
    if not os.path.exists(output_path):
        print(f"\n🆕 No se encontró el archivo Parquet. Iniciando simulación...")
        
        # Simulación con parámetros realistas
        df_parquet = simulator.simulate_dataset(
            num_farmacias=1300,
            num_weeks=52*2,  # 2 años
            start_date=datetime(2024, 1, 1),
            output_path=output_path
        )
        print(f"\n✅ Simulación completada: {output_path}")
    else:
        print(f"\n📦 Archivo Parquet existente encontrado: {output_path}")
        file_size_gb = os.path.getsize(output_path) / (1024**3)
        print(f"   • Tamaño: {file_size_gb:.2f} GB")
    
    # PREPARAR PARA POWER BI (sin cargar en memoria)
    print(f"\n🔄 Preparando datos para Power BI...")
    
    # Preguntar al usuario si quiere generar los CSVs (opcional)
    respuesta = input("¿Generar archivos CSV para Power BI? (s/N): ").strip().lower()
    
    if respuesta == 's':
        # Usar el nuevo método big data
        prepare_for_powerbi_bigdata(
            parquet_path=output_path,
            output_dir=powerbi_dir,
            chunk_size=1000000  # 1M registros por chunk
        )
        print(f"\n✅ Archivos CSV generados en: {powerbi_dir}")
    else:
        print(f"\n⏭️  Omitiendo generación de CSV.")
        print(f"   Puedes usar el archivo Parquet directamente en Power BI:")
        print(f"   • Power BI Desktop > Obtener datos > Parquet")
        print(f"   • Seleccionar: {output_path}")
    
    print(f"\n✨ PROCESO COMPLETADO")

if __name__ == "__main__":
    main()