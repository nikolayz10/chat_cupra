import os
import psycopg2
from psycopg2.extras import RealDictCursor
from openai import OpenAI
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# Configuración PostgreSQL
HOST = os.getenv('HOST')
DBNAME = os.getenv('DBNAME')
USER = os.getenv('USER')
PASSWORD = os.getenv('PASSWORD')
PORT = os.getenv('PORT')
SSLMODE = os.getenv('SSLMODE')

# Configuración OpenAI
OPENAI_API_KEY = os.getenv('KEY_OPENAI')
EMBEDDING_MODEL = os.getenv('MODEL', 'text-embedding-ada-002')

# Configurar cliente OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

class CupraRetrieval:
    """Clase para manejo de búsqueda y recuperación en base de datos CUPRA"""
    
    def __init__(self):
        self.connection_params = {
            'host': HOST,
            'dbname': DBNAME,
            'user': USER,
            'password': PASSWORD,
            'port': PORT,
            'sslmode': SSLMODE
        }
        self._test_connection()
    
    def _test_connection(self):
        """Prueba la conexión a la base de datos"""
        try:
            conn = self._get_connection()
            conn.close()
            # print("✅ Conexión a PostgreSQL exitosa")
        except Exception as e:
            print(f"❌ Error conectando a PostgreSQL: {e}")
            raise
    
    def _get_connection(self):
        """Obtiene una nueva conexión a la base de datos"""
        return psycopg2.connect(**self.connection_params)
    
    def generar_embedding_query(self, query: str, logger=None) -> List[float]:
        """
        Genera embedding para la consulta del usuario
        
        Args:
            query: Texto de la consulta
            
        Returns:
            Lista de floats representando el embedding
        """
        if not query or not query.strip():
            return []
        
        try:
            respuesta = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=query.strip()
            )
            return respuesta.data[0].embedding
            
        except Exception as e:
            # print(f"❌ Error generando embedding para la query: {e}")
            logger.warning(f"❌ Error generando embedding para la query: {e}")
            return []
    
    def buscar_chunks_similares(self, query: str, top_k: int = 4, logger = None) -> List[Dict]:
        """
        Busca los chunks más similares usando búsqueda vectorial coseno
        
        Args:
            query: Consulta de texto del usuario
            top_k: Número de resultados a devolver (default: 4)
            
        Returns:
            Lista de chunks más similares con sus scores
        """
        try:
            # Generar embedding de la consulta
            logger.debug("Convirtiendo pregunta a embedding")
            query_embedding = self.generar_embedding_query(query, logger=None)
            if not query_embedding:
                # print("❌ No se pudo generar embedding para la consulta")
                logger.warning("❌ No se pudo generar embedding para la consulta")
                return []
            
            # Convertir embedding a formato pgvector
            embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
            
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # Consulta SQL con búsqueda vectorial usando similitud coseno
            # 1 - (embedding <=> query) da la similitud coseno (mayor = más similar)
            query_sql = """
            SELECT 
                id,
                title as titulo,
                contenido as cont,
                char_count as num,
                subchunk,
                1 - (embedding <=> %s::vector) as similitud,
                created_at
            FROM cupra_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """
            
            cur.execute(query_sql, (embedding_str, embedding_str, top_k))
            resultados = cur.fetchall()
            
            # Convertir a lista de diccionarios
            chunks_similares = []
            for row in resultados:
                chunk = {
                    'chunk_id': row['id'],
                    'titulo': row['titulo'] or 'Sin título',
                    'cont': row['cont'] or '',
                    'similitud': float(row['similitud']),
                    'subchunk': row['subchunk'] or 0,
                    'num': row['num'] or 0,
                    'created_at': row['created_at']
                }
                chunks_similares.append(chunk)
            
            cur.close()
            conn.close()
            
            # print(f"🔍 Encontrados {len(chunks_similares)} chunks similares")
            # for i, chunk in enumerate(chunks_similares, 1):
            #     similitud_pct = chunk['similitud'] * 100
            #     titulo_short = chunk['titulo'][:50] + "..." if len(chunk['titulo']) > 50 else chunk['titulo']
            #     print(f"   {i}. {titulo_short} (Similitud: {similitud_pct:.1f}%)")
            
            return chunks_similares
            
        except Exception as e:
            # print(f"❌ Error en búsqueda vectorial: {e}")
            logger.error(f"❌ Error en búsqueda vectorial: {e}")
            return []
    
    def buscar_por_titulo(self, titulo_busqueda: str, limit: int = 10) -> List[Dict]:
        """
        Busca chunks por título usando LIKE
        
        Args:
            titulo_busqueda: Texto a buscar en títulos
            limit: Número máximo de resultados
            
        Returns:
            Lista de chunks que coinciden
        """
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
            SELECT 
                id,
                title as titulo,
                contenido as cont,
                char_count as num,
                subchunk,
                created_at
            FROM cupra_chunks
            WHERE title ILIKE %s
            ORDER BY char_count DESC
            LIMIT %s;
            """
            
            cur.execute(query, (f"%{titulo_busqueda}%", limit))
            resultados = cur.fetchall()
            
            chunks = []
            for row in resultados:
                chunk = {
                    'chunk_id': row['id'],
                    'titulo': row['titulo'] or 'Sin título',
                    'cont': row['cont'] or '',
                    'num': row['num'] or 0,
                    'subchunk': row['subchunk'] or 0,
                    'created_at': row['created_at']
                }
                chunks.append(chunk)
            
            cur.close()
            conn.close()
            
            print(f"📚 Encontrados {len(chunks)} chunks por título")
            return chunks
            
        except Exception as e:
            print(f"❌ Error en búsqueda por título: {e}")
            return []
    
    def obtener_estadisticas_bd(self, logger = None) -> Dict[str, Any]:
        """Obtiene estadísticas de la base de datos"""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # Consultas de estadísticas
            queries = {
                'total_chunks': "SELECT COUNT(*) as count FROM cupra_chunks;",
                'titulos_unicos': "SELECT COUNT(DISTINCT title) as count FROM cupra_chunks;",
                'promedio_caracteres': "SELECT AVG(char_count) as avg_chars FROM cupra_chunks;",
                'ultimo_ingreso': "SELECT MAX(created_at) as last_insert FROM cupra_chunks;"
            }
            
            stats = {}
            for key, query in queries.items():
                cur.execute(query)
                result = cur.fetchone()
                stats[key] = result[list(result.keys())[0]]
            
            cur.close()
            conn.close()
            
            return stats
            
        except Exception as e:
            # print(f"❌ Error obteniendo estadísticas: {e}")
            logger.warning(f"Error obteniendo estadísticas: {e}")
            return {}
    
    def verificar_salud_bd(self) -> Dict[str, Any]:
        """Verifica el estado de salud de la base de datos"""
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            
            # Verificar extensión pgvector
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
            pgvector_exists = cur.fetchone() is not None
            
            # Verificar tabla existe
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'cupra_chunks'
                );
            """)
            table_exists = cur.fetchone()[0]
            
            # Verificar índice vectorial
            cur.execute("""
                SELECT indexname FROM pg_indexes 
                WHERE tablename = 'cupra_chunks' 
                AND indexname LIKE '%embedding%';
            """)
            vector_index_exists = cur.fetchone() is not None
            
            cur.close()
            conn.close()
            
            return {
                'pgvector_instalado': pgvector_exists,
                'tabla_existe': table_exists,
                'indice_vectorial': vector_index_exists,
                'conexion_ok': True
            }
            
        except Exception as e:
            print(f"❌ Error verificando salud de BD: {e}")
            return {
                'pgvector_instalado': False,
                'tabla_existe': False,
                'indice_vectorial': False,
                'conexion_ok': False,
                'error': str(e)
            }

# Instancia global del retriever
cupra_retriever = CupraRetrieval()

def busqueda_cupra_chunks(query: str, top_k: int = 4, logger=None) -> List[Dict]:
    """
    Función principal para búsqueda de chunks usando PostgreSQL
    
    Args:
        query: Consulta de texto del usuario
        top_k: Número de resultados (default: 4 como en tu proyecto original)
        
    Returns:
        Lista de chunks relevantes
    """
    try:
        # print(f"🔍 Buscando información para: '{query}'")
        
        # Buscar chunks similares
        resultados = cupra_retriever.buscar_chunks_similares(query, top_k, logger=logger)
        
        if resultados:
            # print(f"✅ Búsqueda exitosa: {len(resultados)} resultados")
            logger.info(f" Búsqueda exitosa: {len(resultados)} resultados")
        else:
            # print("⚠️ No se encontraron resultados")
            logger.warning("⚠️ No se encontraron resultados")
            
        return resultados
        
    except Exception as e:
        print(f"❌ Error en búsqueda de chunks: {e}")
        return []

def mostrar_resultados_busqueda(query: str, resultados: List[Dict]):
    """
    Muestra los resultados de búsqueda de forma detallada
    
    Args:
        query: Consulta original
        resultados: Lista de resultados
    """
    print(f"\n🔍 RESULTADOS PARA: '{query}'")
    print("="*60)
    
    if not resultados:
        print("❌ No se encontraron resultados")
        return
    
    for i, resultado in enumerate(resultados, 1):
        similitud_pct = resultado['similitud'] * 100
        
        print(f"\n🏆 Resultado #{i} - Similitud: {similitud_pct:.1f}%")
        print(f"📋 Título: {resultado['titulo']}")
        print(f"📄 Subchunk: {resultado['subchunk']}")
        print(f"📏 Caracteres: {resultado['num']}")
        print(f"🆔 ID: {resultado['chunk_id']}")
        
        # Mostrar contenido truncado
        contenido = resultado['cont']
        if len(contenido) > 200:
            contenido = contenido[:200] + "..."
        
        print(f"📝 Contenido:\n   {contenido}")
        print("-" * 40)

if __name__ == "__main__":
    """Script de prueba para el sistema de búsqueda"""
    
    print("🚀 SISTEMA DE BÚSQUEDA CUPRA")
    print("="*50)
    
    # Verificar salud del sistema
    salud = cupra_retriever.verificar_salud_bd()
    print("\n🔍 Estado del sistema:")
    for key, value in salud.items():
        status = "✅" if value else "❌"
        print(f"{status} {key}: {value}")
    
    if not salud['conexion_ok']:
        print("❌ No se puede conectar a la base de datos")
        exit(1)
    
    # Mostrar estadísticas
    stats = cupra_retriever.obtener_estadisticas_bd()
    if stats:
        print(f"\n📊 Estadísticas de la base de datos:")
        for key, value in stats.items():
            print(f"   • {key}: {value}")
    
    # Búsqueda interactiva
    print(f"\n💡 Ejemplos de consultas:")
    print("   - 'sistema de luces'")
    print("   - 'airbags de seguridad'")
    print("   - 'climatización'")
    print("   - 'cámara frontal'")
    
    while True:
        try:
            query = input("\n🔍 Ingresa tu consulta (o 'salir' para terminar): ").strip()
            
            if query.lower() in ['salir', 'exit', 'quit', 'q']:
                print("👋 ¡Hasta luego!")
                break
            
            if not query:
                print("⚠️ Por favor ingresa una consulta válida")
                continue
            
            # Realizar búsqueda
            resultados = busqueda_cupra_chunks(query, top_k=4)
            
            # Mostrar resultados
            mostrar_resultados_busqueda(query, resultados)
            
        except KeyboardInterrupt:
            print("\n👋 ¡Búsqueda cancelada!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")