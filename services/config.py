import os
import time
import json
import logging
import logging.handlers as handlers
from sys import stdout, stderr
from datetime import datetime
from typing import Dict, Any

class JSONLogHandler(logging.Handler):
    """
    Handler personalizado que escribe logs en formato JSON con campos especÃ­ficos
    """
    
    def __init__(self, filename: str, max_bytes: int = 500*1024*1024, backup_count: int = 5):
        super().__init__()
        self.filename = filename
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.current_size = 0
        
        # Contadores para cÃ³digos automÃ¡ticos
        self.info_counter = 1
        self.error_counter = 1
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Inicializar tamaÃ±o actual del archivo
        if os.path.exists(filename):
            self.current_size = os.path.getsize(filename)
    
    def emit(self, record):
        """
        Emitir un log record en formato JSON con campos especÃ­ficos
        """
        try:
            # Crear el diccionario del log
            log_entry = self.format_json(record)
            
            # Convertir a JSON y agregar nueva lÃ­nea
            json_line = json.dumps(log_entry, ensure_ascii=False, separators=(',', ':')) + '\n'
            
            # Verificar si necesita rotaciÃ³n
            if self.should_rollover(len(json_line.encode('utf-8'))):
                self.do_rollover()
            
            # Escribir al archivo
            with open(self.filename, 'a', encoding='utf-8') as f:
                f.write(json_line)
                self.current_size += len(json_line.encode('utf-8'))
                
        except Exception:
            self.handleError(record)
    
    def format_json(self, record) -> Dict[str, Any]:
        """
        Formatear el record como diccionario JSON con campos especÃ­ficos
        """
        # Timestamp en formato YYYY-MM-DD HH:MM:SS:SSS
        dt = datetime.fromtimestamp(record.created)
        fechahora = dt.strftime('%Y-%m-%d %H:%M:%S:%f')[:-3]  # Quitar los Ãºltimos 3 dÃ­gitos de microsegundos
        
        # Estructura base para todos los logs
        log_entry = {
            "sistema": "sis",
            "usuario": "user", 
            "fechahora": fechahora,
            "level": record.levelname,
            "descripcion": record.getMessage()
        }        

        return log_entry
    
    def should_rollover(self, message_size: int) -> bool:
        """
        Determinar si el archivo debe rotar
        """
        return self.current_size + message_size >= self.max_bytes
    
    def do_rollover(self):
        """
        Realizar la rotaciÃ³n del archivo
        """
        if os.path.exists(self.filename):
            # Rotar archivos existentes
            for i in range(self.backup_count - 1, 0, -1):
                old_name = f"{self.filename}.{i}"
                new_name = f"{self.filename}.{i + 1}"
                if os.path.exists(old_name):
                    if os.path.exists(new_name):
                        os.remove(new_name)
                    os.rename(old_name, new_name)
            
            # Mover archivo actual
            backup_name = f"{self.filename}.1"
            if os.path.exists(backup_name):
                os.remove(backup_name)
            os.rename(self.filename, backup_name)
            
            # Resetear tamaÃ±o
            self.current_size = 0

class SizedTimedRotatingFileHandler(handlers.TimedRotatingFileHandler):
    """
    Handler para logs que rota por tiempo Y tamaÃ±o
    """
    def __init__(self, filename, maxBytes=0, backupCount=0, encoding="utf-8", 
                 delay=False, when='h', interval=1, utc=False):
        handlers.TimedRotatingFileHandler.__init__(self, filename, when, interval, 
                                                   backupCount, encoding, delay, utc)
        self.maxBytes = maxBytes

    def shouldRollover(self, record):
        if self.stream is None:
            self.stream = self._open()

        if not os.path.exists(self.baseFilename):
            return True

        if self.maxBytes > 0:
            msg = "%s\n" % self.format(record)
            self.stream.seek(0, 2)
            if self.stream.tell() + len(msg) >= self.maxBytes:
                return True

        t = int(time.time())
        if t >= self.rolloverAt:
            return True
            
        return False

def create_rotating_log(path, level, enable_log_file=True, enable_json_file=False):
    """
    Crea un logger con rotaciÃ³n por tiempo y tamaÃ±o, con control independiente para .log y .json
    Configura stdout para INFO y stderr para ERROR segÃºn el estÃ¡ndar
    
    Args:
        path (str): Ruta del archivo de log principal (.log)
        level (str): Nivel de logging
        enable_log_file (bool): Si crear archivos .log
        enable_json_file (bool): Si crear archivos .json
    """
    # Crear directorio si no existe
    log_dir = os.path.dirname(path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configurar logging bÃ¡sico
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.DEBUG,
        handlers=[]
    )

    # Crear logger especÃ­fico
    logger_name = os.path.splitext(os.path.basename(path))[0]
    logger = logging.getLogger(logger_name)
    
    # Limpiar handlers existentes
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Handler para archivo .log (opcional)
    if enable_log_file:
        file_handler = SizedTimedRotatingFileHandler(
            filename=path, 
            when="D", 
            interval=1, 
            backupCount=5, 
            maxBytes=500*(1024**2),
            encoding="utf-8"
        )
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Handler para archivo .json (opcional)
    if enable_json_file:
        json_path = path.replace('.log', '.json')
        json_handler = JSONLogHandler(
            filename=json_path,
            max_bytes=500*(1024**2),
            backup_count=5
        )
        logger.addHandler(json_handler)
    
    # Handler para stdout (INFO y DEBUG)
    stdout_handler = logging.StreamHandler(stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stdout_handler.setFormatter(stdout_formatter)
    
    # Filtro para que stdout solo reciba INFO y DEBUG
    class StdoutFilter(logging.Filter):
        def filter(self, record):
            return record.levelno <= logging.INFO
    
    stdout_handler.addFilter(StdoutFilter())
    logger.addHandler(stdout_handler)
    
    # Handler para stderr (WARNING, ERROR, CRITICAL)
    stderr_handler = logging.StreamHandler(stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stderr_handler.setFormatter(stderr_formatter)
    logger.addHandler(stderr_handler)
    
    # Configurar nivel de logging
    nivel = level.lower()
    if nivel == "debug":
        logger.setLevel(logging.DEBUG)
    elif nivel == "info":
        logger.setLevel(logging.INFO)
    elif nivel == "warning":
        logger.setLevel(logging.WARNING)
    elif nivel == "error":
        logger.setLevel(logging.ERROR)
    elif nivel == "fatal":
        logger.setLevel(logging.FATAL)
    elif nivel == "critical":
        logger.setLevel(logging.CRITICAL)
    else:
        logger.setLevel(logging.INFO)
    
    logger.propagate = False
    
    return logger

# ConfiguraciÃ³n global de logs
class LogConfig:
    def __init__(self):
        self.logs_folder = "./logs"
        self.level_log = "info"
        self.enable_log_files = False      # Control para archivos .log
        self.enable_json_files = False     # Control para archivos .json
    
    def set_logs_folder(self, folder):
        self.logs_folder = folder
    
    def set_log_level(self, level):
        self.level_log = level
    
    def set_log_files(self, enable):
        """Habilitar/deshabilitar archivos .log"""
        self.enable_log_files = enable
    
    def set_json_files(self, enable):
        """Habilitar/deshabilitar archivos .json"""
        self.enable_json_files = enable

# Instancia global de configuraciÃ³n
log_config = LogConfig()