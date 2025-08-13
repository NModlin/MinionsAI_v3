"""
MinionsAI v3.1 - Advanced Tool Ecosystem
Comprehensive tool management system with custom tool framework and specialized tools.
"""

import os
import json
import sqlite3
import requests
import asyncio
from typing import Dict, List, Any, Optional, Callable, Union, Type
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
from pathlib import Path
import logging
import importlib.util
import inspect

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class ToolMetadata:
    """Metadata for a tool."""
    name: str
    description: str
    category: str
    version: str = "1.0.0"
    author: str = "MinionsAI"
    tags: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    is_enabled: bool = True
    usage_count: int = 0
    last_used: Optional[datetime] = None
    performance_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "version": self.version,
            "author": self.author,
            "tags": self.tags,
            "requirements": self.requirements,
            "is_enabled": self.is_enabled,
            "usage_count": self.usage_count,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "performance_score": self.performance_score
        }


class AdvancedBaseTool(BaseTool, ABC):
    """
    Enhanced base tool with advanced features.
    """
    
    metadata: ToolMetadata = Field(default_factory=lambda: ToolMetadata("", "", ""))
    
    def __init__(self, **kwargs):
        """Initialize the advanced tool."""
        super().__init__(**kwargs)
        if not self.metadata.name:
            self.metadata.name = self.name
        if not self.metadata.description:
            self.metadata.description = self.description
    
    async def arun(self, *args, **kwargs) -> Any:
        """Async run with performance tracking."""
        start_time = datetime.now()
        try:
            result = await self._arun(*args, **kwargs)
            self._update_performance_metrics(True, (datetime.now() - start_time).total_seconds())
            return result
        except Exception as e:
            self._update_performance_metrics(False, (datetime.now() - start_time).total_seconds())
            raise e
    
    def _run(self, *args, **kwargs) -> Any:
        """Sync run with performance tracking."""
        start_time = datetime.now()
        try:
            result = self._execute(*args, **kwargs)
            self._update_performance_metrics(True, (datetime.now() - start_time).total_seconds())
            return result
        except Exception as e:
            self._update_performance_metrics(False, (datetime.now() - start_time).total_seconds())
            raise e
    
    @abstractmethod
    def _execute(self, *args, **kwargs) -> Any:
        """Execute the tool's main functionality."""
        pass
    
    async def _arun(self, *args, **kwargs) -> Any:
        """Async execution (default to sync)."""
        return self._execute(*args, **kwargs)
    
    def _update_performance_metrics(self, success: bool, execution_time: float) -> None:
        """Update tool performance metrics."""
        self.metadata.usage_count += 1
        self.metadata.last_used = datetime.now()
        
        # Simple performance scoring
        if success:
            # Reward fast execution
            time_score = max(0.1, 1.0 - (execution_time / 10.0))  # Normalize to 10 seconds
            self.metadata.performance_score = (
                self.metadata.performance_score * 0.9 + time_score * 0.1
            )
        else:
            # Penalize failures
            self.metadata.performance_score *= 0.95


class FileOperationsTool(AdvancedBaseTool):
    """Advanced file operations tool."""

    name: str = "file_operations"
    description: str = "Perform file and directory operations including read, write, list, and manage files"

    def __init__(self):
        super().__init__()
        self.metadata = ToolMetadata(
            name=self.name,
            description=self.description,
            category="file_system",
            tags=["files", "directories", "io"]
        )
        # Set as instance attributes instead of class attributes
        self.allowed_extensions = {'.txt', '.json', '.csv', '.md', '.py', '.js', '.html', '.css'}
        self.max_file_size = 10 * 1024 * 1024  # 10MB
    
    def _execute(self, operation: str, path: str, content: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Execute file operation."""
        try:
            if operation == "read":
                return self._read_file(path)
            elif operation == "write":
                return self._write_file(path, content or "")
            elif operation == "list":
                return self._list_directory(path)
            elif operation == "delete":
                return self._delete_file(path)
            elif operation == "exists":
                return {"exists": os.path.exists(path)}
            elif operation == "info":
                return self._get_file_info(path)
            else:
                return {"error": f"Unknown operation: {operation}"}
        
        except Exception as e:
            return {"error": str(e)}
    
    def _read_file(self, path: str) -> Dict[str, Any]:
        """Read file content."""
        file_path = Path(path)
        
        if not file_path.exists():
            return {"error": "File not found"}
        
        if file_path.suffix.lower() not in self.allowed_extensions:
            return {"error": f"File type not allowed: {file_path.suffix}"}
        
        if file_path.stat().st_size > self.max_file_size:
            return {"error": "File too large"}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return {
                "content": content,
                "size": len(content),
                "path": str(file_path),
                "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            }
        
        except UnicodeDecodeError:
            return {"error": "File is not text-readable"}
    
    def _write_file(self, path: str, content: str) -> Dict[str, Any]:
        """Write content to file."""
        file_path = Path(path)
        
        if file_path.suffix.lower() not in self.allowed_extensions:
            return {"error": f"File type not allowed: {file_path.suffix}"}
        
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return {
                "success": True,
                "path": str(file_path),
                "size": len(content),
                "modified": datetime.now().isoformat()
            }
        
        except Exception as e:
            return {"error": f"Failed to write file: {str(e)}"}
    
    def _list_directory(self, path: str) -> Dict[str, Any]:
        """List directory contents."""
        dir_path = Path(path)
        
        if not dir_path.exists():
            return {"error": "Directory not found"}
        
        if not dir_path.is_dir():
            return {"error": "Path is not a directory"}
        
        try:
            items = []
            for item in dir_path.iterdir():
                item_info = {
                    "name": item.name,
                    "path": str(item),
                    "is_directory": item.is_dir(),
                    "size": item.stat().st_size if item.is_file() else 0,
                    "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                }
                items.append(item_info)
            
            return {
                "items": sorted(items, key=lambda x: (not x["is_directory"], x["name"])),
                "count": len(items),
                "path": str(dir_path)
            }
        
        except Exception as e:
            return {"error": f"Failed to list directory: {str(e)}"}
    
    def _delete_file(self, path: str) -> Dict[str, Any]:
        """Delete a file."""
        file_path = Path(path)
        
        if not file_path.exists():
            return {"error": "File not found"}
        
        try:
            if file_path.is_file():
                file_path.unlink()
            elif file_path.is_dir():
                file_path.rmdir()  # Only remove empty directories
            
            return {"success": True, "path": str(file_path)}
        
        except Exception as e:
            return {"error": f"Failed to delete: {str(e)}"}
    
    def _get_file_info(self, path: str) -> Dict[str, Any]:
        """Get file information."""
        file_path = Path(path)
        
        if not file_path.exists():
            return {"error": "File not found"}
        
        try:
            stat = file_path.stat()
            return {
                "path": str(file_path),
                "name": file_path.name,
                "size": stat.st_size,
                "is_directory": file_path.is_dir(),
                "is_file": file_path.is_file(),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "extension": file_path.suffix.lower()
            }
        
        except Exception as e:
            return {"error": f"Failed to get file info: {str(e)}"}


class DatabaseTool(AdvancedBaseTool):
    """Database connectivity and operations tool."""
    
    name: str = "database"
    description: str = "Connect to and query databases (SQLite, PostgreSQL, MySQL)"
    
    def __init__(self):
        super().__init__()
        self.metadata = ToolMetadata(
            name=self.name,
            description=self.description,
            category="database",
            tags=["database", "sql", "data"]
        )
        self.connections: Dict[str, Any] = {}
    
    def _execute(self, operation: str, connection_string: str, query: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Execute database operation."""
        try:
            if operation == "connect":
                return self._connect(connection_string)
            elif operation == "query":
                return self._query(connection_string, query or "")
            elif operation == "execute":
                return self._execute_query(connection_string, query or "")
            elif operation == "tables":
                return self._list_tables(connection_string)
            elif operation == "schema":
                return self._get_schema(connection_string, kwargs.get("table_name"))
            else:
                return {"error": f"Unknown operation: {operation}"}
        
        except Exception as e:
            return {"error": str(e)}
    
    def _connect(self, connection_string: str) -> Dict[str, Any]:
        """Connect to database."""
        try:
            # For now, only support SQLite
            if connection_string.startswith("sqlite://"):
                db_path = connection_string.replace("sqlite://", "")
                conn = sqlite3.connect(db_path)
                conn.row_factory = sqlite3.Row  # Enable column access by name
                
                self.connections[connection_string] = conn
                
                return {
                    "success": True,
                    "connection_string": connection_string,
                    "database_type": "sqlite"
                }
            else:
                return {"error": "Only SQLite databases are currently supported"}
        
        except Exception as e:
            return {"error": f"Failed to connect: {str(e)}"}
    
    def _query(self, connection_string: str, query: str) -> Dict[str, Any]:
        """Execute a SELECT query."""
        if connection_string not in self.connections:
            connect_result = self._connect(connection_string)
            if "error" in connect_result:
                return connect_result
        
        try:
            conn = self.connections[connection_string]
            cursor = conn.cursor()
            cursor.execute(query)
            
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            # Convert rows to dictionaries
            results = []
            for row in rows:
                results.append(dict(zip(columns, row)))
            
            return {
                "results": results,
                "row_count": len(results),
                "columns": columns,
                "query": query
            }
        
        except Exception as e:
            return {"error": f"Query failed: {str(e)}"}
    
    def _execute_query(self, connection_string: str, query: str) -> Dict[str, Any]:
        """Execute an INSERT/UPDATE/DELETE query."""
        if connection_string not in self.connections:
            connect_result = self._connect(connection_string)
            if "error" in connect_result:
                return connect_result
        
        try:
            conn = self.connections[connection_string]
            cursor = conn.cursor()
            cursor.execute(query)
            conn.commit()
            
            return {
                "success": True,
                "rows_affected": cursor.rowcount,
                "query": query
            }
        
        except Exception as e:
            return {"error": f"Query execution failed: {str(e)}"}
    
    def _list_tables(self, connection_string: str) -> Dict[str, Any]:
        """List all tables in the database."""
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        result = self._query(connection_string, query)
        
        if "error" in result:
            return result
        
        tables = [row["name"] for row in result["results"]]
        return {"tables": tables, "count": len(tables)}
    
    def _get_schema(self, connection_string: str, table_name: Optional[str]) -> Dict[str, Any]:
        """Get schema information for a table."""
        if not table_name:
            return {"error": "Table name is required"}
        
        query = f"PRAGMA table_info({table_name});"
        result = self._query(connection_string, query)
        
        if "error" in result:
            return result
        
        schema = []
        for row in result["results"]:
            schema.append({
                "column": row["name"],
                "type": row["type"],
                "nullable": not row["notnull"],
                "default": row["dflt_value"],
                "primary_key": bool(row["pk"])
            })
        
        return {"table": table_name, "schema": schema}


class APIIntegrationTool(AdvancedBaseTool):
    """API integration and HTTP request tool."""
    
    name: str = "api_integration"
    description: str = "Make HTTP requests to APIs and web services"
    
    def __init__(self):
        super().__init__()
        self.metadata = ToolMetadata(
            name=self.name,
            description=self.description,
            category="integration",
            tags=["api", "http", "web", "integration"]
        )
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "MinionsAI/3.1"})
    
    def _execute(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Execute HTTP request."""
        try:
            # Security check - only allow certain domains or require explicit approval
            allowed_domains = kwargs.get("allowed_domains", [])
            if allowed_domains and not any(domain in url for domain in allowed_domains):
                return {"error": "Domain not in allowed list"}
            
            # Prepare request parameters
            headers = kwargs.get("headers", {})
            params = kwargs.get("params", {})
            data = kwargs.get("data")
            json_data = kwargs.get("json")
            timeout = kwargs.get("timeout", 30)
            
            # Make request
            response = self.session.request(
                method=method.upper(),
                url=url,
                headers=headers,
                params=params,
                data=data,
                json=json_data,
                timeout=timeout
            )
            
            # Parse response
            result = {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "url": response.url,
                "method": method.upper()
            }
            
            # Try to parse JSON, fall back to text
            try:
                result["data"] = response.json()
                result["content_type"] = "json"
            except:
                result["data"] = response.text
                result["content_type"] = "text"
            
            return result
        
        except requests.exceptions.Timeout:
            return {"error": "Request timed out"}
        except requests.exceptions.ConnectionError:
            return {"error": "Connection error"}
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}


class CustomToolFramework:
    """
    Framework for creating and managing custom tools.
    """
    
    def __init__(self, tools_directory: str = "custom_tools"):
        """Initialize the custom tool framework."""
        self.tools_directory = Path(tools_directory)
        self.tools_directory.mkdir(exist_ok=True)
        self.custom_tools: Dict[str, Type[AdvancedBaseTool]] = {}
        self.tool_instances: Dict[str, AdvancedBaseTool] = {}
    
    def create_tool_template(self, tool_name: str) -> str:
        """Create a template for a custom tool."""
        template = f'''"""
Custom tool: {tool_name}
Generated by MinionsAI Custom Tool Framework
"""

from core.advanced.tools import AdvancedBaseTool, ToolMetadata
from typing import Dict, Any, Optional


class {tool_name.title().replace("_", "")}Tool(AdvancedBaseTool):
    """Custom tool: {tool_name}"""
    
    name: str = "{tool_name}"
    description: str = "Custom tool for {tool_name} operations"
    
    def __init__(self):
        super().__init__()
        self.metadata = ToolMetadata(
            name=self.name,
            description=self.description,
            category="custom",
            tags=["custom", "{tool_name}"]
        )
    
    def _execute(self, *args, **kwargs) -> Dict[str, Any]:
        """Execute the tool's main functionality."""
        # TODO: Implement your tool logic here
        return {{"result": "Tool {tool_name} executed successfully"}}
'''
        
        tool_file = self.tools_directory / f"{tool_name}.py"
        with open(tool_file, 'w') as f:
            f.write(template)
        
        return str(tool_file)
    
    def load_custom_tools(self) -> Dict[str, bool]:
        """Load all custom tools from the tools directory."""
        results = {}
        
        for tool_file in self.tools_directory.glob("*.py"):
            if tool_file.name.startswith("__"):
                continue
            
            try:
                # Load module
                spec = importlib.util.spec_from_file_location(tool_file.stem, tool_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find tool classes
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, AdvancedBaseTool) and 
                        obj != AdvancedBaseTool):
                        
                        self.custom_tools[obj.name] = obj
                        results[obj.name] = True
                        logger.info(f"Loaded custom tool: {obj.name}")
            
            except Exception as e:
                logger.error(f"Failed to load tool from {tool_file}: {e}")
                results[tool_file.stem] = False
        
        return results
    
    def get_tool_instance(self, tool_name: str) -> Optional[AdvancedBaseTool]:
        """Get an instance of a custom tool."""
        if tool_name not in self.tool_instances:
            if tool_name in self.custom_tools:
                self.tool_instances[tool_name] = self.custom_tools[tool_name]()
            else:
                return None
        
        return self.tool_instances[tool_name]


class AdvancedToolManager:
    """
    Comprehensive tool management system.
    """
    
    def __init__(self):
        """Initialize the advanced tool manager."""
        self.tools: Dict[str, AdvancedBaseTool] = {}
        self.custom_framework = CustomToolFramework()
        self.tool_categories: Dict[str, List[str]] = {}
        
        # Initialize built-in tools
        self._initialize_builtin_tools()
        
        # Load custom tools
        self.custom_framework.load_custom_tools()
    
    def _initialize_builtin_tools(self) -> None:
        """Initialize built-in advanced tools."""
        builtin_tools = [
            FileOperationsTool(),
            DatabaseTool(),
            APIIntegrationTool()
        ]
        
        for tool in builtin_tools:
            self.register_tool(tool)
    
    def register_tool(self, tool: AdvancedBaseTool) -> None:
        """Register a tool."""
        self.tools[tool.name] = tool
        
        # Organize by category
        category = tool.metadata.category
        if category not in self.tool_categories:
            self.tool_categories[category] = []
        
        if tool.name not in self.tool_categories[category]:
            self.tool_categories[category].append(tool.name)
        
        logger.info(f"Registered tool: {tool.name} in category: {category}")
    
    def get_tool(self, tool_name: str) -> Optional[AdvancedBaseTool]:
        """Get a tool by name."""
        # Check built-in tools first
        if tool_name in self.tools:
            return self.tools[tool_name]
        
        # Check custom tools
        return self.custom_framework.get_tool_instance(tool_name)
    
    def list_tools(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available tools."""
        tools_list = []
        
        for tool_name, tool in self.tools.items():
            if category is None or tool.metadata.category == category:
                tools_list.append(tool.metadata.to_dict())
        
        # Add custom tools
        for tool_name, tool_class in self.custom_framework.custom_tools.items():
            tool_instance = self.custom_framework.get_tool_instance(tool_name)
            if tool_instance and (category is None or tool_instance.metadata.category == category):
                tools_list.append(tool_instance.metadata.to_dict())
        
        return sorted(tools_list, key=lambda x: x["name"])
    
    def get_tool_categories(self) -> Dict[str, List[str]]:
        """Get tools organized by category."""
        return self.tool_categories.copy()
    
    def execute_tool(self, tool_name: str, *args, **kwargs) -> Dict[str, Any]:
        """Execute a tool."""
        tool = self.get_tool(tool_name)
        if not tool:
            return {"error": f"Tool '{tool_name}' not found"}
        
        if not tool.metadata.is_enabled:
            return {"error": f"Tool '{tool_name}' is disabled"}
        
        try:
            result = tool._run(*args, **kwargs)
            return {"success": True, "result": result, "tool": tool_name}
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {"error": f"Tool execution failed: {str(e)}", "tool": tool_name}
    
    async def execute_tool_async(self, tool_name: str, *args, **kwargs) -> Dict[str, Any]:
        """Execute a tool asynchronously."""
        tool = self.get_tool(tool_name)
        if not tool:
            return {"error": f"Tool '{tool_name}' not found"}
        
        if not tool.metadata.is_enabled:
            return {"error": f"Tool '{tool_name}' is disabled"}
        
        try:
            result = await tool.arun(*args, **kwargs)
            return {"success": True, "result": result, "tool": tool_name}
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {"error": f"Tool execution failed: {str(e)}", "tool": tool_name}
    
    def get_tool_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics."""
        total_tools = len(self.tools) + len(self.custom_framework.custom_tools)
        enabled_tools = sum(1 for tool in self.tools.values() if tool.metadata.is_enabled)
        
        category_counts = {}
        for category, tools in self.tool_categories.items():
            category_counts[category] = len(tools)
        
        return {
            "total_tools": total_tools,
            "builtin_tools": len(self.tools),
            "custom_tools": len(self.custom_framework.custom_tools),
            "enabled_tools": enabled_tools,
            "categories": category_counts
        }
