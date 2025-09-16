from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, Any
import os
from datetime import datetime
from services.config import create_rotating_log, log_config

# Importar el pipeline CUPRA
from services.llm.cupra_rag_pipeline import CupraRAGPipeline
from services.llm.cupra_retrieval import cupra_retriever

app = FastAPI(
    title="CUPRA Assistant API",
    description="API para asistencia inteligente de vehículos CUPRA",
    version="1.0.0"
)

# Montar archivos estáticos (para chatbot.html)
# app.mount("/static", StaticFiles(directory="static"), name="images/logo_cupra.png")  #revisarlo
# app.mount("/static", StaticFiles(directory="static"), name="static")

#-----------------------------------------------------------------------------------------------------
logger = None

def initialize():
    global logger

    # Configurar directorio de logs
    log_config.set_logs_folder("./logs")
    log_config.set_log_level("info")
    guardar_log = False
    
    if guardar_log:
        # Crear directorio de logs si no existe
        if not os.path.exists(log_config.logs_folder):
            os.makedirs(log_config.logs_folder)
    
    # Inicializar logger
    nombreLog = os.path.splitext(os.path.basename(__file__))[0]  # "main"
    log_path = os.path.join(log_config.logs_folder, f"{nombreLog}.log")
    
    logger = create_rotating_log(
        path=log_path, 
        level=log_config.level_log,
        enable_log_file=guardar_log,
        enable_json_file=guardar_log
    )
    
    return logger

# Inicializar al importar
logger = initialize()
#-----------------------------------------------------------------------------------------------------

# Inicializar pipeline
pipeline = None

try:
    pipeline = CupraRAGPipeline(logger=logger)
    # logger.info(" CUPRA RAG Pipeline inicializado correctamente")
except Exception as e:
    logger.warning(f"❌ Error inicializando pipeline: {e}")
    pipeline = None

# Modelos Pydantic
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    success: bool
    query: str
    respuesta_llm: Dict[str, Any]
    evaluacion_calidad: str
    chunks_recuperados: int
    timestamp: str
    fuente: str

class HealthResponse(BaseModel):
    status: str
    pipeline_available: bool
    database_status: Dict[str, Any]
    timestamp: str

# Rutas
@app.get("/", response_class=HTMLResponse)
async def chatbot_interface():
    """Servir la interfaz del chatbot"""
    try:
        # with open("chatbot.html", "r", encoding="utf-8") as f:
        # with open("templates/SS_SS.html", "r", encoding="utf-8") as f:
        with open("templates/Ss.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        logger.error(" .html no encontrado")
        return HTMLResponse(
            content="<h1>Error: chatbot.html no encontrado</h1>",
            status_code=404
        )

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint principal para procesar consultas del chatbot
    
    Args:
        request: Objeto con la consulta del usuario
        
    Returns:
        Respuesta completa del pipeline RAG
    """
    try:
        # Verificar que el pipeline esté disponible
        # if pipeline is None:
        #     raise HTTPException(
        #         status_code=500, 
        #         detail="Pipeline no disponible - Error en la inicialización del sistema"
        #     )
        if pipeline is None:
            return JSONResponse(
                status_code=503,
                content={
                    "success": False,
                    "query": request.query,
                    "respuesta_llm": {"respuesta": "Servicio en modo degradado (DB no disponible).", "confianza": 0.0, "fuentes": []},
                    "evaluacion_calidad": "5",
                    "chunks_recuperados": 0,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "fuente": "PostgreSQL"
                }
            )

        # Validar consulta
        query = request.query.strip()
        if not query:
            raise HTTPException(
                status_code=400, 
                detail="La consulta no puede estar vacía"
            )

        logger.info(f" Nueva consulta recibida: {query}")

        # Procesar consulta usando el pipeline completo
        resultado = pipeline.procesar_consulta_completa(query)
        
        # Preparar respuesta
        respuesta = ChatResponse(
            success=True,
            query=query,
            respuesta_llm={
                'respuesta': resultado['respuesta_llm']['respuesta'],
                'confianza': resultado['respuesta_llm']['confianza'],
                'fuentes': resultado['respuesta_llm']['fuentes']
            },
            evaluacion_calidad=resultado['evaluacion_calidad'],
            chunks_recuperados=len(resultado['chunks_recuperados']),
            timestamp=resultado['timestamp'],
            fuente='PostgreSQL'
        )
        
        # print(f"✅ Consulta procesada exitosamente - Confianza: {resultado['respuesta_llm']['confianza']:.2f}")
        logger.info(f"Consulta procesada exitosamente - Confianza: {resultado['respuesta_llm']['confianza']:.2f}")
        
        return respuesta

    except HTTPException:
        raise
    except Exception as e:
        # print(f"❌ Error procesando consulta: {e}")
        logger.warning(f"❌ Error procesando consulta: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error interno procesando la consulta: {str(e)}"
        )

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    Endpoint para verificar el estado del sistema
    
    Returns:
        Estado del pipeline y base de datos
    """
    try:
        # Verificar estado de la base de datos
        salud_bd = cupra_retriever.verificar_salud_bd() if cupra_retriever else {
        "pgvector_instalado": False,
        "tabla_existe": False,
        "indice_vectorial": False,
        "conexion_ok": False,
        "error": "Retriever no disponible al arranque"
        }
        
        # salud_bd = cupra_retriever.verificar_salud_bd()
        
        # Determinar estado general
        status = "ok" if pipeline is not None and salud_bd['conexion_ok'] else "error"
        
        return HealthResponse(
            status=status,
            pipeline_available=pipeline is not None,
            database_status=salud_bd,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
    except Exception as e:
        # print(f"❌ Error en health check: {e}")
        logger.warning(f"❌ Error en health check: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error verificando estado del sistema: {str(e)}"
        )

@app.get("/api/database/stats")
async def database_stats():
    """
    Endpoint para obtener estadísticas de la base de datos
    
    Returns:
        Estadísticas detalladas de la base de datos
    """
    try:
        stats = cupra_retriever.obtener_estadisticas_bd()
        salud = cupra_retriever.verificar_salud_bd()
        
        return {
            "success": True,
            "statistics": stats,
            "health": salud,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        # print(f"❌ Error obteniendo estadísticas: {e}")
        logger.warning(f"❌ Error obteniendo estadísticas: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error obteniendo estadísticas: {str(e)}"
        )

@app.post("/api/search/title")
async def search_by_title(titulo: str, limit: int = 10):
    """
    Endpoint para buscar chunks por título específico
    
    Args:
        titulo: Texto a buscar en títulos
        limit: Número máximo de resultados
        
    Returns:
        Lista de chunks que coinciden con el título
    """
    try:
        if not titulo.strip():
            raise HTTPException(
                status_code=400, 
                detail="El título no puede estar vacío"
            )

        # Buscar por título
        resultados = cupra_retriever.buscar_por_titulo(titulo.strip(), limit)
        
        return {
            "success": True,
            "titulo_buscado": titulo,
            "resultados": resultados,
            "total_encontrados": len(resultados),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        # print(f"❌ Error en búsqueda por título: {e}")
        logger.warning(f"❌ Error en búsqueda por título: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error en la búsqueda: {str(e)}"
        )

@app.get("/api/examples")
async def get_examples():
    """
    Endpoint para obtener ejemplos de consultas
    
    Returns:
        Lista de consultas de ejemplo
    """
    ejemplos = [
        "¿Cómo funciona el sistema de luces del CUPRA?",
        "¿Qué tipos de airbags tiene el vehículo?",
        "¿Cómo se usa la climatización?",
        "¿Cuáles son las características de la cámara frontal?",
        "¿Cómo configurar el sistema de navegación?",
        "¿Qué sistemas de seguridad incluye el vehículo?",
        "¿Cómo funciona el sistema de frenado?",
        "¿Cuáles son las características del motor?",
        "¿Cómo se ajustan los asientos?",
        "¿Qué hacer si aparece una luz de advertencia?"
    ]
    
    return {
        "success": True,
        "ejemplos": ejemplos,
        "total": len(ejemplos)
    }

# Manejo de errores globales
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint no encontrado",
            "message": "La ruta solicitada no existe"
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Error interno del servidor",
            "message": "Ha ocurrido un error interno"
        }
    )

# Eventos de ciclo de vida
@app.on_event("startup")
async def startup_event():
    """Eventos al iniciar la aplicación"""
    # print("🚀 CUPRA Assistant API iniciada")
    logger.info("CUPRA Assistant API iniciada")
    
    # if pipeline:
    #     print("✅ Pipeline RAG disponible")
    # else:
    #     print("❌ Pipeline RAG no disponible")
        
    if not pipeline:
        logger.error(" Pipeline RAG no disponible")
    
    # Mostrar estadísticas iniciales
    try:
        # stats = cupra_retriever.obtener_estadisticas_bd(logger = logger)
        # print(f"📊 Base de datos: {stats.get('total_chunks', 0)} chunks disponibles")
        # logger.info(f"Base de datos: {stats.get('total_chunks', 0)} chunks disponibles")
        logger.info("Base de datos: creo que iniciada")
    except:
        # print("⚠️ No se pudieron obtener estadísticas de la base de datos")
        logger.warning("⚠️ No se pudieron obtener estadísticas de la base de datos")

@app.on_event("shutdown")
async def shutdown_event():
    """Eventos al cerrar la aplicación"""
    # print("🛑 CUPRA Assistant API detenida")
    logger.info("CUPRA Assistant API detenida")

# Ejecutar la aplicación
if __name__ == "__main__":
    import uvicorn
    
    # Verificar configuración antes de iniciar
    required_vars = ['KEY_OPENAI', 'HOST', 'DBNAME', 'USER', 'PASSWORD']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        # print(f"❌ Variables de entorno faltantes: {missing_vars}")
        logger.error(f"❌ Variables de entorno faltantes: {missing_vars}")
        exit(1)
    
    # Verificar estado del sistema
    if pipeline is None:
        print("⚠️ Advertencia: Pipeline no disponible - algunas funcionalidades pueden no funcionar")
        logger.warning("⚠️ Advertencia: Pipeline no disponible - algunas funcionalidades pueden no funcionar")
    
    # print("🌐 Iniciando servidor FastAPI...")
    # print("📱 Chatbot disponible en: http://localhost:8000")
    # print("📖 Documentación API: http://localhost:8000/docs")
    print("Chatbot disponible en: http://localhost:8000")
    
    # Iniciar servidor
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False  # Para desarrollo
    )