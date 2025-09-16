import json
import os
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
from services.llm.cupra_retrieval import busqueda_cupra_chunks, cupra_retriever

load_dotenv()

# Configuración
OPENAI_API_KEY = os.getenv('KEY_OPENAI')
MODEL_GPT = os.getenv('MODEL_LLM', 'gpt-4o-mini')

# Configurar cliente OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

class CupraRAGPipeline:
    """Pipeline RAG completo para asistencia CUPRA: RAG → LLM → Quality Agent"""
    
    def __init__(self, logger=None):
        """Inicializa el pipeline verificando conexiones"""
        # print("🚀 Inicializando CUPRA RAG Pipeline...")
        
        self.logger = logger
        
        # Verificar conexión a base de datos
        salud_bd = cupra_retriever.verificar_salud_bd()
        if not salud_bd['conexion_ok']:
            # self.logger.error("❌ No se puede conectar a la base de datos PostgreSQL")
            raise Exception("❌ No se puede conectar a la base de datos PostgreSQL")
        
        # Verificar que hay datos
        stats = cupra_retriever.obtener_estadisticas_bd()
        if stats.get('total_chunks', 0) == 0:
            # self.logger.error("❌ No hay chunks en la base de datos")
            raise Exception("❌ No hay chunks en la base de datos")
        
        # self.logger.info(f"✅ Pipeline inicializado - {stats.get('total_chunks', 0)} chunks disponibles")
    
    def paso_1_rag(self, query: str, top_k: int = 4) -> List[Dict]:
        """
        PASO 1: RAG - Recuperación de información relevante
        
        Args:
            query: Consulta del usuario
            top_k: Número de chunks a recuperar (default: 4)
            
        Returns:
            Lista de chunks más relevantes con similitud coseno
        """
        # print(f"🔍 PASO 1 - RAG: Buscando información para '{query}'...")
        self.logger.info(f"PASO 1 - RAG: Buscando información para '{query}'...")
        
        try:
            # Usar el sistema de búsqueda vectorial
            resultados = busqueda_cupra_chunks(query, top_k=top_k, logger = self.logger)
            
            if resultados:
                # print(f"✅ RAG exitoso: {len(resultados)} chunks recuperados")
                self.logger.info(f" RAG exitoso: {len(resultados)} chunks recuperados")
                # for i, resultado in enumerate(resultados, 1):
                #     similitud_pct = resultado['similitud'] * 100
                #     titulo_short = resultado['titulo'][:50] + "..." if len(resultado['titulo']) > 50 else resultado['titulo']
                #     print(f"   {i}. {titulo_short} ({similitud_pct:.1f}%)")
            else:
                # print("❌ RAG: No se encontraron chunks relevantes")
                self.logger.error("❌ RAG: No se encontraron chunks relevantes")
                
            return resultados
            
        except Exception as e:
            # print(f"❌ Error en PASO 1 - RAG: {e}")
            self.logger.warning(f"❌ Error en PASO 1 - RAG: {e}")
            return []
    
    def paso_2_llm(self, query: str, chunks_relevantes: List[Dict]) -> Dict[str, Any]:
        """
        PASO 2: LLM - Generación de respuesta con contexto
        
        Args:
            query: Consulta original del usuario
            chunks_relevantes: Chunks recuperados del RAG
            
        Returns:
            Diccionario con la respuesta generada y metadatos
        """
        # print(f"🤖 PASO 2 - LLM: Generando respuesta...")
        self.logger.info(f"PASO 2 - LLM: Generando respuesta...")
        
        if not chunks_relevantes:
            self.logger.error("No se encontro información relevante en el manual CUPRA")
            return {
                'respuesta': "Lo siento, no encontré información relevante en el manual de CUPRA para responder tu consulta. ¿Podrías reformular tu pregunta o ser más específico?",
                'contexto_usado': [],
                'confianza': 0.0,
                'fuentes': []
            }
        
        try:
            # Construir contexto desde los chunks
            self.logger.debug("Construyendo contexto desde los chunks")
            contexto = self._construir_contexto(chunks_relevantes)
            
            # Crear prompt especializado para CUPRA
            self.logger.debug("Creando prompt")
            prompt = self._crear_prompt_cupra(query, contexto)
            
            # Llamar al LLM
            self.logger.debug("haciendo llamada al llm")
            respuesta = client.chat.completions.create(
                model=MODEL_GPT,
                messages=[
                    {
                        "role": "system", 
                        "content": "Eres un asistente técnico especializado en vehículos CUPRA. Responde ÚNICAMENTE basándote en la información del manual oficial proporcionada. Si no tienes información suficiente, indícalo claramente. Mantén un tono profesional pero accesible."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Baja temperatura para respuestas más consistentes
                max_tokens=1000
            )
            
            respuesta_texto = respuesta.choices[0].message.content
            
            # Calcular confianza promedio basada en similitud de chunks
            confianza = sum(chunk['similitud'] for chunk in chunks_relevantes) / len(chunks_relevantes)
            
            # Extraer información de fuentes
            fuentes = [
                {
                    'titulo': chunk['titulo'],
                    'similitud': chunk['similitud'],
                    'subchunk': chunk['subchunk'],
                    'chunk_id': chunk['chunk_id']
                }
                for chunk in chunks_relevantes
            ]
            
            resultado = {
                'respuesta': respuesta_texto,
                'contexto_usado': contexto,
                'confianza': confianza,
                'fuentes': fuentes
            }
            
            # print(f"✅ LLM: Respuesta generada (Confianza: {confianza:.2f})")
            self.logger.info(f" LLM Confianza: {confianza:.2f}")
            return resultado
            
        except Exception as e:
            # print(f"❌ Error en PASO 2 - LLM: {e}")
            self.logger.warning(f"❌ Error en PASO 2 - LLM: {e}")
            return {
                'respuesta': f"Error generando respuesta: {str(e)}",
                'contexto_usado': [],
                'confianza': 0.0,
                'fuentes': []
            }
    
    def paso_3_quality_agent(self, query: str, respuesta_llm: Dict[str, Any]) -> str:
        """
        PASO 3: Quality Agent - Evaluación de calidad de la respuesta
        
        Args:
            query: Consulta original
            respuesta_llm: Respuesta del LLM con metadatos
            
        Returns:
            Puntuación de calidad (1-10)
        """
        # print(f"🔍 PASO 3 - Quality Agent: Evaluando calidad...")
        self.logger.info(f"PASO 3 - Quality Agent: Evaluando calidad...")
        
        try:
            # Crear prompt para evaluación de calidad
            prompt_quality = self._crear_prompt_quality(query, respuesta_llm)
            
            # Evaluar calidad
            evaluacion = client.chat.completions.create(
                model=MODEL_GPT,
                messages=[
                    {
                        "role": "system", 
                        "content": "Eres un evaluador de calidad de respuestas técnicas. Evalúa la respuesta del 1 al 10 considerando: relevancia, precisión, completitud, claridad y fundamentación. Responde SOLO con el número."
                    },
                    {
                        "role": "user", 
                        "content": prompt_quality
                    }
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            evaluacion_texto = evaluacion.choices[0].message.content.strip()
            
            # Extraer número de la evaluación
            try:
                puntuacion = int(evaluacion_texto)
                if 1 <= puntuacion <= 10:
                    # print(f"✅ Quality Agent: Puntuación {puntuacion}/10")
                    self.logger.info(f"Quality Agent: Puntuación {puntuacion}/10")
                    return str(puntuacion)
                else:
                    # print(f"⚠️ Quality Agent: Puntuación fuera de rango, usando 5")
                    self.logger.error(f"⚠️ Quality Agent: Puntuación fuera de rango, usando 5")
                    return "5"
            except:
                # print(f"⚠️ Quality Agent: No se pudo extraer puntuación, usando 5")
                self.logger.error(f"⚠️ Quality Agent: No se pudo extraer puntuación, usando 5")
                return "5"
            
        except Exception as e:
            print(f"❌ Error en PASO 3 - Quality Agent: {e}")
            self.logger.warning(f"Error en PASO 3 - Quality Agent: {e}")
            return "5"  # Puntuación por defecto
    
    def _construir_contexto(self, chunks: List[Dict]) -> List[str]:
        """Construye el contexto a partir de los chunks relevantes"""
        contexto_partes = []
        
        for i, chunk in enumerate(chunks, 1):
            contexto_parte = f"INFORMACIÓN {i}:\n"
            contexto_parte += f"Título: {chunk['titulo']}\n"
            if chunk['subchunk'] > 0:
                contexto_parte += f"Sección: {chunk['subchunk']}\n"
            contexto_parte += f"Contenido: {chunk['cont']}\n"
            contexto_partes.append(contexto_parte)
        
        return contexto_partes
    
    def _crear_prompt_cupra(self, query: str, contexto: List[str]) -> str:
        """Crea el prompt especializado para CUPRA"""
        contexto_str = "\n\n".join(contexto)
        
        prompt = f"""Basándote EXCLUSIVAMENTE en la siguiente información del manual oficial de CUPRA, responde la consulta del usuario de manera precisa y útil.

INFORMACIÓN DEL MANUAL CUPRA:
{contexto_str}

CONSULTA DEL USUARIO:
{query}

INSTRUCCIONES DE FORMATO Y CONTENIDO:
1. Responde ÚNICAMENTE basándote en la información proporcionada del manual
2. IMPORTANTE - Estructura tu respuesta de forma clara y legible:
   - Usa SALTOS DE LÍNEA entre secciones principales
   - Para pasos numerados, pon cada paso en una nueva línea
   - Separa claramente las secciones (ej: pasos principales, condiciones, precauciones)
   - Usa **negrita** para títulos de secciones importantes
   - Deja una línea en blanco entre párrafos diferentes
3. Si hay procedimientos paso a paso:
   - Numera cada paso principal (1., 2., 3., etc.)
   - Los sub-pasos van con guiones (-)
   - Cada paso en su propia línea
4. Agrupa la información en secciones lógicas como:
   - Pasos principales
   - Condiciones importantes
   - Precauciones o advertencias
   - Información adicional
5. Mantén un tono profesional pero accesible
6. Si la información no es suficiente, indícalo claramente al final

FORMATO DE EJEMPLO para respuestas con pasos:

**Título de la función**

**Pasos principales:**

1. **Primer paso**
   - Detalle del paso
   - Otro detalle si es necesario

2. **Segundo paso**
   - Explicación clara

**Condiciones importantes:**
- Primera condición
- Segunda condición

**Precauciones:**
⚠️ Advertencia importante

RESPUESTA:"""
        
        return prompt
    
    def _crear_prompt_quality(self, query: str, respuesta_llm: Dict[str, Any]) -> str:
        """Crea el prompt para evaluación de calidad"""
        prompt = f"""Evalúa del 1 al 10 la calidad de esta respuesta sobre vehículos CUPRA:

CONSULTA:
{query}

RESPUESTA:
{respuesta_llm['respuesta']}

CRITERIOS DE EVALUACIÓN:
1. Relevancia: ¿Responde directamente a la consulta?
2. Precisión: ¿Es técnicamente correcta?
3. Completitud: ¿Cubre los aspectos importantes?
4. Claridad: ¿Es fácil de entender?
5. Fundamentación: ¿Está basada en la información proporcionada?

Responde SOLO con el número del 1 al 10:"""
        
        return prompt
    
    def procesar_consulta_completa(self, query: str) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo: RAG → LLM → Quality Agent
        
        Args:
            query: Consulta del usuario
            
        Returns:
            Diccionario con todos los resultados del pipeline
        """
        # print(f"\n{'='*60}")
        # print(f"🚀 CUPRA RAG PIPELINE INICIADO")
        # print(f"🔍 Consulta: '{query}'")
        # print(f"{'='*60}")
        
        self.logger.info("CUPRA RAG PIPELINE INICIADO")
        
        # Paso 1: RAG - Recuperación
        chunks_relevantes = self.paso_1_rag(query, top_k=4)
        
        # Paso 2: LLM - Generación
        respuesta_llm = self.paso_2_llm(query, chunks_relevantes)
        
        # Paso 3: Quality Agent - Evaluación
        evaluacion_calidad = self.paso_3_quality_agent(query, respuesta_llm)
        
        # Resultado final
        resultado_final = {
            'query': query,
            'chunks_recuperados': chunks_relevantes,
            'respuesta_llm': respuesta_llm,
            'evaluacion_calidad': evaluacion_calidad,
            'timestamp': self._get_timestamp(),
            'fuente': 'PostgreSQL'
        }
        
        # Mostrar resumen
        # self._mostrar_resumen_final(resultado_final)
        
        return resultado_final
    
    def _get_timestamp(self) -> str:
        """Obtiene timestamp actual"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _mostrar_resumen_final(self, resultado: Dict[str, Any]):
        """Muestra un resumen del procesamiento completo"""
        print(f"\n{'='*60}")
        print(f"📋 RESUMEN FINAL - CUPRA RAG PIPELINE")
        print(f"{'='*60}")
        
        print(f"📊 Chunks recuperados: {len(resultado['chunks_recuperados'])}")
        print(f"🤖 Respuesta generada: {len(resultado['respuesta_llm']['respuesta'])} caracteres")
        print(f"⭐ Confianza: {resultado['respuesta_llm']['confianza']:.2f}")
        print(f"🏆 Puntuación calidad: {resultado['evaluacion_calidad']}/10")
        print(f"💾 Fuente: {resultado['fuente']}")
        
        print(f"\n🔍 RESPUESTA FINAL:")
        print(f"{'─'*40}")
        print(resultado['respuesta_llm']['respuesta'])
        print(f"{'─'*40}")

def main():
    """Función principal para probar el pipeline"""
    try:
        # Verificar configuración
        if not OPENAI_API_KEY:
            print("❌ Error: Configura tu API key de OpenAI")
            return
        
        # Inicializar pipeline
        pipeline = CupraRAGPipeline()
        
        print("🚀 CUPRA RAG Pipeline iniciado")
        print("💡 Ejemplos de consultas:")
        print("   - '¿Cómo funciona el sistema de luces del CUPRA?'")
        print("   - '¿Qué tipos de airbags tiene el vehículo?'")
        print("   - '¿Cómo se usa la climatización?'")
        print("   - '¿Cuáles son las características de la cámara frontal?'")
        
        while True:
            # Obtener consulta del usuario
            query = input("\n🔍 Ingresa tu consulta (o 'salir' para terminar): ").strip()
            
            if query.lower() in ['salir', 'exit', 'quit', 'q']:
                print("👋 ¡Hasta luego!")
                break
            
            if not query:
                print("⚠️ Por favor ingresa una consulta válida")
                continue
            
            # Procesar consulta completa
            resultado = pipeline.procesar_consulta_completa(query)
            
            # Opcional: Guardar resultado
            with open("resultado_cupra.json", 'w', encoding='utf-8') as f:
                json.dump(resultado, f, ensure_ascii=False, indent=2)
            
    except KeyboardInterrupt:
        print("\n👋 ¡Proceso cancelado!")
    except Exception as e:
        print(f"❌ Error en el proceso principal: {e}")

if __name__ == "__main__":
    main()