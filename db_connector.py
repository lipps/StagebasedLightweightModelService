"""
数据库连接适配器
支持 MySQL 和 SQL Server
"""
import os
import logging
from typing import Optional, Any, Dict
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# 数据库类型配置
DB_TYPE = os.environ.get('DB_TYPE', 'mysql').lower()  # mysql 或 mssql

# SQL Server 配置（默认指向 PhonecallQuality 库）
MSSQL_HOST = os.environ.get('MSSQL_HOST', '192.168.24.101')
MSSQL_PORT = int(os.environ.get('MSSQL_PORT', 1433))
MSSQL_USER = os.environ.get('MSSQL_USER', 'AIUser')
MSSQL_PASSWORD = os.environ.get('MSSQL_PASSWORD', 'hOe9xvd#CcvM')
MSSQL_DATABASE = os.environ.get('MSSQL_DATABASE', 'PhonecallQuality')
MSSQL_DRIVER = os.environ.get('MSSQL_DRIVER', 'ODBC Driver 17 for SQL Server')

# MySQL 配置（兼容旧配置）
MYSQL_HOST = os.environ.get('DB_HOST', '192.168.42.13')
MYSQL_PORT = int(os.environ.get('DB_PORT', 3306))
MYSQL_USER = os.environ.get('DB_USER', 'root')
MYSQL_PASSWORD = os.environ.get('DB_PASSWORD', '')
MYSQL_DATABASE = os.environ.get('DB_DATABASE', 'ai_project')


class DatabaseConnection:
    """统一数据库连接管理器"""

    def __init__(self, db_type: Optional[str] = None):
        self.db_type = (db_type or DB_TYPE).lower()
        self.connection = None

    def __enter__(self):
        if self.db_type == 'mssql':
            return self._connect_mssql()
        elif self.db_type == 'mysql':
            return self._connect_mysql()
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            self.connection.close()
            logger.debug(f"Database connection closed ({self.db_type})")
        return False

    def _connect_mssql(self):
        """连接 SQL Server"""
        try:
            import pymssql

            self.connection = pymssql.connect(
                server=MSSQL_HOST,
                port=MSSQL_PORT,
                user=MSSQL_USER,
                password=MSSQL_PASSWORD,
                database=MSSQL_DATABASE,
                charset='UTF-8',
                as_dict=True,
                timeout=30
            )
            logger.debug(f"SQL Server connected: {MSSQL_HOST}:{MSSQL_PORT}/{MSSQL_DATABASE}")
            return self.connection

        except ImportError:
            logger.error("pymssql not installed. Run: pip install pymssql")
            raise
        except Exception as e:
            logger.error(f"SQL Server connection failed: {e}")
            raise

    def _connect_mysql(self):
        """连接 MySQL（兼容旧代码）"""
        try:
            import pymysql

            self.connection = pymysql.connect(
                host=MYSQL_HOST,
                port=MYSQL_PORT,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                database=MYSQL_DATABASE,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor,
                connect_timeout=10,
                read_timeout=30,
                write_timeout=30
            )
            logger.debug(f"MySQL connected: {MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}")
            return self.connection

        except Exception as e:
            logger.error(f"MySQL connection failed: {e}")
            raise


@contextmanager
def get_db_connection(db_type: Optional[str] = None):
    """获取数据库连接的上下文管理器"""
    conn_manager = DatabaseConnection(db_type)
    conn = conn_manager.__enter__()
    try:
        yield conn
    finally:
        conn_manager.__exit__(None, None, None)


# 测试连接函数
def test_connection(db_type: Optional[str] = None) -> bool:
    """
    测试数据库连接

    Returns:
        bool: 连接成功返回 True，失败返回 False
    """
    try:
        with get_db_connection(db_type) as conn:
            cursor = conn.cursor()

            # 执行简单查询测试
            if (db_type or DB_TYPE) == 'mssql':
                cursor.execute("SELECT @@VERSION AS version")
            else:
                cursor.execute("SELECT VERSION() AS version")

            result = cursor.fetchone()
            logger.info(f"Database connection test successful: {result}")
            return True

    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


# ============================================================================
# Transcript数据库连接器（阶段一：证据跳转功能）
# ============================================================================

import sqlite3
import uuid
import hashlib
import json
from datetime import datetime
from typing import List, Tuple


class TranscriptDatabaseConnector:
    """
    通话原文和分析结果数据库连接器

    支持双模式：
    - SQLite: 本地存储（批处理、开发测试）
    - MySQL: 集中存储（生产环境、多实例部署）

    使用场景：
    - API服务：写入MySQL（主）
    - 批处理脚本：写入SQLite（主） + 同步MySQL（副）
    - 历史查询：优先查MySQL，回退SQLite

    参考：todoAddEvidences.md 第7.4节
    """

    def __init__(
        self,
        sqlite_path: str = "data/analysis_results.sqlite3",
        mysql_config: Optional[Dict[str, Any]] = None,
        mode: str = "sqlite"  # "sqlite" | "mysql" | "both"
    ):
        """
        初始化数据库连接器

        Args:
            sqlite_path: SQLite数据库文件路径
            mysql_config: MySQL配置字典 {host, port, user, password, database}
            mode: 运行模式
                - "sqlite": 仅使用SQLite
                - "mysql": 仅使用MySQL
                - "both": 双写模式（SQLite主，MySQL同步）
        """
        self.sqlite_path = sqlite_path
        self.mysql_config = mysql_config or {
            "host": MYSQL_HOST,
            "port": MYSQL_PORT,
            "user": MYSQL_USER,
            "password": MYSQL_PASSWORD,
            "database": MYSQL_DATABASE
        }
        self.mode = mode
        self._sqlite_conn = None
        self._mysql_conn = None

        # 统计信息（用于监控）
        self._stats = {
            "insert_count": 0,
            "query_count": 0,
            "mysql_insert_count": 0,
            "mysql_query_count": 0,
            "mysql_error_count": 0,
            "sqlite_insert_count": 0,
            "sqlite_query_count": 0,
            "sqlite_error_count": 0,
            "start_time": datetime.now().isoformat()
        }

        # 自动创建表
        if self.mode in ["sqlite", "both"]:
            self._ensure_sqlite_tables()
        if self.mode in ["mysql", "both"]:
            self._ensure_mysql_tables()

    def _ensure_sqlite_tables(self):
        """确保SQLite表存在"""
        try:
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()

            # 创建 call_transcripts 表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS call_transcripts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transcript_uid TEXT NOT NULL UNIQUE,
                    source_filename TEXT NULL,
                    source_hash TEXT NULL,
                    original_transcript TEXT NOT NULL,
                    processed_dialogues TEXT NOT NULL,
                    total_utterances INTEGER NOT NULL DEFAULT 0,
                    has_timestamps INTEGER NOT NULL DEFAULT 0,
                    schema_version INTEGER NOT NULL DEFAULT 1,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 创建索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_transcript_created_at ON call_transcripts(created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_transcript_source ON call_transcripts(source_filename)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_transcript_hash ON call_transcripts(source_hash)')

            # 创建 call_analysis_results 表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS call_analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_filename TEXT NULL,
                    transcript_id INTEGER NULL,
                    schema_version INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'pending',
                    analysis_result TEXT NULL,
                    error_message TEXT NULL,
                    processed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (transcript_id)
                        REFERENCES call_transcripts(id)
                        ON DELETE SET NULL
                )
            ''')

            # 迁移逻辑：为旧表添加新列（如果不存在）
            # 检查 call_analysis_results 表是否有 transcript_id 列
            cursor.execute("PRAGMA table_info(call_analysis_results)")
            columns = [row[1] for row in cursor.fetchall()]

            if 'transcript_id' not in columns:
                logger.info("迁移: 添加 transcript_id 列到 call_analysis_results")
                cursor.execute('ALTER TABLE call_analysis_results ADD COLUMN transcript_id INTEGER NULL')

            if 'schema_version' not in columns:
                logger.info("迁移: 添加 schema_version 列到 call_analysis_results")
                cursor.execute('ALTER TABLE call_analysis_results ADD COLUMN schema_version INTEGER NOT NULL DEFAULT 1')

            # 创建索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_result_transcript_id ON call_analysis_results(transcript_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_result_status ON call_analysis_results(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_result_processed_at ON call_analysis_results(processed_at)')

            conn.commit()
            conn.close()
            logger.info(f"SQLite tables ensured at {self.sqlite_path}")
        except Exception as e:
            logger.error(f"Failed to ensure SQLite tables: {e}")
            raise

    def _get_sqlite_connection(self) -> sqlite3.Connection:
        """获取SQLite连接"""
        if not self._sqlite_conn:
            self._sqlite_conn = sqlite3.connect(self.sqlite_path)
            self._sqlite_conn.row_factory = sqlite3.Row  # 返回字典
        return self._sqlite_conn

    def _get_mysql_connection(self):
        """获取MySQL连接（支持连接池）"""
        # 尝试使用连接池（如果已初始化）
        try:
            from src.utils.connection_pool import get_global_pool

            # 获取全局连接池
            pool = get_global_pool(
                host=self.mysql_config["host"],
                port=self.mysql_config["port"],
                user=self.mysql_config["user"],
                password=self.mysql_config["password"],
                database=self.mysql_config["database"]
            )

            # 返回连接池连接（注意：这里返回池，实际使用时用 with pool.get_connection()）
            # 为了保持向后兼容，这里仍然返回单个连接
            if not self._mysql_conn:
                import pymysql
                self._mysql_conn = pymysql.connect(
                    host=self.mysql_config["host"],
                    port=self.mysql_config["port"],
                    user=self.mysql_config["user"],
                    password=self.mysql_config["password"],
                    database=self.mysql_config["database"],
                    charset='utf8mb4',
                    cursorclass=pymysql.cursors.DictCursor
                )
            return self._mysql_conn

        except ImportError:
            # 连接池不可用，回退到普通连接
            logger.debug("连接池不可用，使用普通 MySQL 连接")
            if not self._mysql_conn:
                import pymysql
                self._mysql_conn = pymysql.connect(
                    host=self.mysql_config["host"],
                    port=self.mysql_config["port"],
                    user=self.mysql_config["user"],
                    password=self.mysql_config["password"],
                    database=self.mysql_config["database"],
                    charset='utf8mb4',
                    cursorclass=pymysql.cursors.DictCursor
                )
            return self._mysql_conn

    def _ensure_mysql_tables(self):
        """确保MySQL表存在（含迁移逻辑）"""
        try:
            import pymysql
            conn = pymysql.connect(
                host=self.mysql_config["host"],
                port=self.mysql_config["port"],
                user=self.mysql_config["user"],
                password=self.mysql_config["password"],
                database=self.mysql_config["database"],
                charset='utf8mb4'
            )
            cursor = conn.cursor()

            # 创建 call_transcripts 表（MySQL语法）
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS call_transcripts (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    transcript_uid VARCHAR(36) NOT NULL UNIQUE,
                    source_filename VARCHAR(255) NULL,
                    source_hash VARCHAR(64) NULL,
                    original_transcript LONGTEXT NOT NULL,
                    processed_dialogues LONGTEXT NOT NULL,
                    total_utterances INT NOT NULL DEFAULT 0,
                    has_timestamps TINYINT NOT NULL DEFAULT 0,
                    schema_version INT NOT NULL DEFAULT 1,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_transcript_created_at (created_at),
                    INDEX idx_transcript_source (source_filename),
                    INDEX idx_transcript_hash (source_hash)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            ''')

            # 创建 call_analysis_results 表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS call_analysis_results (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    source_filename VARCHAR(255) NULL,
                    transcript_id INT NULL,
                    schema_version INT NOT NULL DEFAULT 1,
                    status VARCHAR(20) NOT NULL DEFAULT 'pending',
                    analysis_result LONGTEXT NULL,
                    error_message TEXT NULL,
                    processed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_result_transcript_id (transcript_id),
                    INDEX idx_result_status (status),
                    INDEX idx_result_processed_at (processed_at),
                    FOREIGN KEY (transcript_id)
                        REFERENCES call_transcripts(id)
                        ON DELETE SET NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            ''')

            # 迁移逻辑：检查列是否存在（MySQL 5.7+ 支持）
            # 查询表结构
            cursor.execute("SHOW COLUMNS FROM call_analysis_results LIKE 'transcript_id'")
            if not cursor.fetchone():
                logger.info("迁移: 添加 transcript_id 列到 MySQL call_analysis_results")
                cursor.execute('ALTER TABLE call_analysis_results ADD COLUMN transcript_id INT NULL')

            cursor.execute("SHOW COLUMNS FROM call_analysis_results LIKE 'schema_version'")
            if not cursor.fetchone():
                logger.info("迁移: 添加 schema_version 列到 MySQL call_analysis_results")
                cursor.execute('ALTER TABLE call_analysis_results ADD COLUMN schema_version INT NOT NULL DEFAULT 1')

            conn.commit()
            conn.close()
            logger.info(f"MySQL tables ensured at {self.mysql_config['host']}:{self.mysql_config['database']}")

        except Exception as e:
            logger.error(f"Failed to ensure MySQL tables: {e}")
            if self.mode == "mysql":
                raise
            else:
                logger.warning("MySQL table creation failed, continuing with SQLite only")

    def insert_transcript(
        self,
        original_transcript: str,
        processed_dialogues: List[Dict[str, Any]],
        source_filename: Optional[str] = None,
        schema_version: int = 1
    ) -> Tuple[int, str]:
        """
        插入通话原文数据

        Args:
            original_transcript: 原始通话文本
            processed_dialogues: 结构化对话列表
            source_filename: 源文件名（可选）
            schema_version: processed_text版本号

        Returns:
            Tuple[int, str]: (transcript_id, transcript_uid)
        """
        # 生成UUID和哈希
        transcript_uid = str(uuid.uuid4())
        source_hash = hashlib.sha256(original_transcript.encode('utf-8')).hexdigest()

        # 序列化dialogues为JSON
        processed_dialogues_json = json.dumps(processed_dialogues, ensure_ascii=False)

        # 统计元数据
        total_utterances = len(processed_dialogues)
        has_timestamps = any('timestamp' in d and d['timestamp'] for d in processed_dialogues)

        if self.mode in ["sqlite", "both"]:
            # 插入SQLite
            try:
                conn = self._get_sqlite_connection()
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO call_transcripts (
                        transcript_uid, source_filename, source_hash,
                        original_transcript, processed_dialogues,
                        total_utterances, has_timestamps, schema_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    transcript_uid, source_filename, source_hash,
                    original_transcript, processed_dialogues_json,
                    total_utterances, int(has_timestamps), schema_version
                ))
                conn.commit()
                transcript_id_sqlite = cursor.lastrowid
                self._stats["sqlite_insert_count"] += 1
                logger.info(f"Transcript inserted to SQLite: id={transcript_id_sqlite}, uid={transcript_uid}")
            except Exception as e:
                self._stats["sqlite_error_count"] += 1
                logger.error(f"Failed to insert transcript to SQLite: {e}")
                if self.mode == "sqlite":
                    raise

        if self.mode in ["mysql", "both"]:
            # 插入MySQL
            try:
                conn = self._get_mysql_connection()
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO call_transcripts (
                        transcript_uid, source_filename, source_hash,
                        original_transcript, processed_dialogues,
                        total_utterances, has_timestamps, schema_version
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ''', (
                    transcript_uid, source_filename, source_hash,
                    original_transcript, processed_dialogues_json,
                    total_utterances, int(has_timestamps), schema_version
                ))
                conn.commit()
                transcript_id_mysql = cursor.lastrowid
                self._stats["mysql_insert_count"] += 1
                logger.info(f"Transcript inserted to MySQL: id={transcript_id_mysql}, uid={transcript_uid}")
            except Exception as e:
                self._stats["mysql_error_count"] += 1
                logger.error(f"Failed to insert transcript to MySQL: {e}")
                if self.mode == "mysql":
                    raise

        # 更新总计数
        self._stats["insert_count"] += 1

        # 返回SQLite的ID（或MySQL的ID，取决于模式）
        transcript_id = transcript_id_sqlite if self.mode != "mysql" else transcript_id_mysql
        return transcript_id, transcript_uid

    def get_transcript(self, transcript_id: int) -> Optional[Dict[str, Any]]:
        """
        根据ID查询原文

        Args:
            transcript_id: 原文记录ID

        Returns:
            Dict or None: 原文数据字典
        """
        self._stats["query_count"] += 1

        # 优先查MySQL，失败回退SQLite
        if self.mode in ["mysql", "both"]:
            try:
                conn = self._get_mysql_connection()
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM call_transcripts WHERE id = %s', (transcript_id,))
                row = cursor.fetchone()
                if row:
                    self._stats["mysql_query_count"] += 1
                    row['processed_dialogues'] = json.loads(row['processed_dialogues'])
                    return dict(row)
            except Exception as e:
                self._stats["mysql_error_count"] += 1
                logger.warning(f"Failed to query MySQL, fallback to SQLite: {e}")

        if self.mode in ["sqlite", "both"]:
            try:
                conn = self._get_sqlite_connection()
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM call_transcripts WHERE id = ?', (transcript_id,))
                row = cursor.fetchone()
                if row:
                    self._stats["sqlite_query_count"] += 1
                    result = dict(row)
                    result['processed_dialogues'] = json.loads(result['processed_dialogues'])
                    return result
            except Exception as e:
                self._stats["sqlite_error_count"] += 1
                logger.error(f"Failed to query SQLite: {e}")

        return None

    def get_transcript_by_uid(self, transcript_uid: str) -> Optional[Dict[str, Any]]:
        """根据UUID查询原文"""
        self._stats["query_count"] += 1

        if self.mode in ["mysql", "both"]:
            try:
                conn = self._get_mysql_connection()
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM call_transcripts WHERE transcript_uid = %s', (transcript_uid,))
                row = cursor.fetchone()
                if row:
                    row['processed_dialogues'] = json.loads(row['processed_dialogues'])
                    self._stats["mysql_query_count"] += 1
                    return dict(row)
            except Exception as e:
                logger.warning(f"Failed to query MySQL by UID, fallback to SQLite: {e}")
                self._stats["mysql_error_count"] += 1

        if self.mode in ["sqlite", "both"]:
            try:
                conn = self._get_sqlite_connection()
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM call_transcripts WHERE transcript_uid = ?', (transcript_uid,))
                row = cursor.fetchone()
                if row:
                    result = dict(row)
                    result['processed_dialogues'] = json.loads(result['processed_dialogues'])
                    self._stats["sqlite_query_count"] += 1
                    return result
            except Exception as e:
                logger.error(f"Failed to query SQLite by UID: {e}")
                self._stats["sqlite_error_count"] += 1

        return None

    def insert_analysis_result(
        self,
        transcript_id: int,
        analysis_result_json: str,
        source_filename: Optional[str] = None,
        schema_version: int = 1,
        status: str = "completed"
    ) -> int:
        """
        插入分析结果

        Args:
            transcript_id: 关联的原文ID
            analysis_result_json: 分析结果JSON字符串
            source_filename: 源文件名
            schema_version: 结果schema版本
            status: 分析状态

        Returns:
            int: 分析结果ID
        """
        if self.mode in ["sqlite", "both"]:
            conn = self._get_sqlite_connection()
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO call_analysis_results (
                    source_filename, transcript_id, schema_version,
                    status, analysis_result
                ) VALUES (?, ?, ?, ?, ?)
            ''', (source_filename, transcript_id, schema_version, status, analysis_result_json))
            conn.commit()
            result_id_sqlite = cursor.lastrowid
            logger.info(f"Analysis result inserted to SQLite: id={result_id_sqlite}")

        if self.mode in ["mysql", "both"]:
            try:
                conn = self._get_mysql_connection()
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO call_analysis_results (
                        source_filename, transcript_id, schema_version,
                        status, analysis_result
                    ) VALUES (%s, %s, %s, %s, %s)
                ''', (source_filename, transcript_id, schema_version, status, analysis_result_json))
                conn.commit()
                result_id_mysql = cursor.lastrowid
                logger.info(f"Analysis result inserted to MySQL: id={result_id_mysql}")
            except Exception as e:
                logger.error(f"Failed to insert analysis result to MySQL: {e}")
                if self.mode == "mysql":
                    raise

        result_id = result_id_sqlite if self.mode != "mysql" else result_id_mysql
        return result_id

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据库连接器统计信息

        Returns:
            Dict containing:
            - insert_count: 总插入次数
            - query_count: 总查询次数
            - mysql_insert_count: MySQL插入次数
            - mysql_query_count: MySQL查询次数
            - mysql_error_count: MySQL错误次数
            - sqlite_insert_count: SQLite插入次数
            - sqlite_query_count: SQLite查询次数
            - sqlite_error_count: SQLite错误次数
            - start_time: 启动时间
            - uptime_seconds: 运行时长（秒）
            - total_operations: 总操作次数
            - total_errors: 总错误次数
            - error_rate_percent: 错误率（百分比）
            - mode: 数据库模式
        """
        from datetime import datetime

        start_dt = datetime.fromisoformat(self._stats["start_time"])
        uptime = (datetime.now() - start_dt).total_seconds()

        total_ops = self._stats["insert_count"] + self._stats["query_count"]
        total_errors = self._stats["mysql_error_count"] + self._stats["sqlite_error_count"]
        error_rate = (total_errors / total_ops * 100) if total_ops > 0 else 0.0

        return {
            **self._stats,
            "uptime_seconds": round(uptime, 2),
            "total_operations": total_ops,
            "total_errors": total_errors,
            "error_rate_percent": round(error_rate, 2),
            "mode": self.mode
        }

    def close(self):
        """关闭所有连接"""
        if self._sqlite_conn:
            self._sqlite_conn.close()
            self._sqlite_conn = None
        if self._mysql_conn:
            self._mysql_conn.close()
            self._mysql_conn = None


def export_call_records_task():
    """导出最近100条通话记录内容"""
    output_dir = "data_history"
    limit = 100
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created directory: {output_dir}")

    # Explicitly use [PhonecallQuality].[dbo].[call_record] as requested
    query = f"SELECT TOP {limit} call_id, call_content FROM [PhonecallQuality].[dbo].[call_record]"
    
    try:
        # Force MSSQL connection
        with get_db_connection('mssql') as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            
            rows = cursor.fetchall()
            logger.info(f"Fetched {len(rows)} records.")
            
            for row in rows:
                call_id = row['call_id']
                content = row['call_content']
                
                # Check if call_id exists
                if call_id:
                    file_path = os.path.join(output_dir, f"{call_id}.json")
                    with open(file_path, 'w', encoding='utf-8') as f:
                        # Save only call_content field content
                        json.dump({"call_content": content}, f, ensure_ascii=False, indent=2)
                        
            logger.info(f"Export completed. Files saved to {output_dir}")
            return True

    except Exception as e:
        logger.error(f"Export task failed: {e}")
        return False


if __name__ == "__main__":
    # 命令行测试与数据导出
    import sys
    logging.basicConfig(level=logging.INFO)

    test_type = sys.argv[1] if len(sys.argv) > 1 else 'mssql'
    
    print(f"Testing connection for {test_type}...")
    success = test_connection(test_type)

    if success:
        print("✓ Database connection successful!")
        
        # Execute export task if testing mssql
        if test_type == 'mssql':
            print("Starting data export task (Top 100 call_records)...")
            if export_call_records_task():
                print("✓ Data export successful!")
                sys.exit(0)
            else:
                print("✗ Data export failed!")
                sys.exit(1)
        else:
             sys.exit(0)
    else:
        print("✗ Database connection failed!")
        sys.exit(1)

