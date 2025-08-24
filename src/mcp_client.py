"""
MCP (Model Context Protocol) client management and tool retrieval.

This module handles the setup and management of MCP clients,
server configurations, and tool retrieval.
"""

import asyncio
import atexit
import logging
import random
from typing import Dict, List, Optional
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

# Module-level singleton instance
_connection_manager_instance: Optional['MCPConnectionManager'] = None


class MCPConnectionManager:
    """
    Singleton manager for MCP connections with heartbeat, reconnection, and tool caching.
    
    Features:
    - Singleton pattern with thread-safe initialization
    - Automatic heartbeat to keep SSE connections alive
    - Exponential backoff reconnection with jitter
    - Tool caching with refresh capabilities
    - Proper async resource cleanup
    """
    
    def __init__(self):
        self.client: Optional[MultiServerMCPClient] = None
        self.server_config: Optional[Dict[str, Dict]] = None
        self.lock = asyncio.Lock()
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.reconnect_task: Optional[asyncio.Task] = None
        self.running = False
        self.tools_cache: List[BaseTool] = []
        self.last_heartbeat_ok = False  # Start as False until first successful connection
        self._shutdown_registered = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
    
    @classmethod
    def get_instance(cls) -> 'MCPConnectionManager':
        """Get or create the singleton instance."""
        global _connection_manager_instance
        if _connection_manager_instance is None:
            _connection_manager_instance = cls()
            # Register cleanup on process exit
            if not _connection_manager_instance._shutdown_registered:
                atexit.register(_connection_manager_instance._cleanup_on_exit)
                _connection_manager_instance._shutdown_registered = True
        return _connection_manager_instance
    
    def _cleanup_on_exit(self):
        """Cleanup method for atexit handler."""
        try:
            if self.running:
                # Prefer using the original loop that created tasks
                loop = self._loop
                if loop and not loop.is_closed():
                    if loop.is_running():
                        try:
                            fut = asyncio.run_coroutine_threadsafe(self.close(), loop)
                            fut.result(timeout=2)
                        except Exception as e:
                            logger.error(f"Error during MCP connection cleanup (thread-safe): {e}")
                    else:
                        loop.run_until_complete(self.close())
                else:
                    logger.debug("Skipping MCP cleanup: no active event loop available")
        except Exception as e:
            logger.error(f"Error during MCP connection cleanup: {e}")
    
    async def start(self, server_config: Dict[str, Dict]) -> None:
        """
        Start the MCP connection with the given configuration.
        
        Args:
            server_config: Server configuration dictionary
        """
        async with self.lock:
            # If already running with same config, no-op
            if self.running and self.server_config == server_config:
                logger.info("MCP connection already running with same configuration")
                return
            
            # If running but config changed, close and restart
            if self.running:
                logger.info("Configuration changed, restarting MCP connection")
                await self._close_internal()
            
            # Store config and create client
            self.server_config = server_config
            try:
                self.client = MultiServerMCPClient(server_config)
                
                # Validate connectivity early
                await self.client.get_tools()
                logger.info("MCP client successfully connected and validated")
                
                # Mark as successfully connected
                self.last_heartbeat_ok = True
                self.running = True
                
                # Clear tools cache to force refresh on next get_tools call
                self.tools_cache = []
                
                # Start heartbeat task
                self._loop = asyncio.get_running_loop()
                self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                
            except Exception as e:
                logger.error(f"Failed to start MCP connection: {e}")
                await self._close_internal()
                raise
    
    async def get_tools(self, force_refresh: bool = False) -> List[BaseTool]:
        """
        Get tools from the MCP client, with caching.
        
        Args:
            force_refresh: If True, bypass cache and fetch fresh tools
            
        Returns:
            List of available tools
        """
        if not self.client:
            logger.warning("No MCP client available")
            return []
        
        if not self.tools_cache or force_refresh:
            try:
                logger.debug("Fetching tools from MCP client...")
                self.tools_cache = await self.client.get_tools()
                logger.info(f"Retrieved {len(self.tools_cache)} tools from MCP server")
            except Exception as e:
                logger.error(f"Failed to get tools from MCP client: {e}")
                if not self.tools_cache:  # Only return empty if no cached tools
                    return []
        else:
            logger.debug(f"Using cached tools: {len(self.tools_cache)} tools")
        
        return self.tools_cache.copy()
    
    async def close(self) -> None:
        """Close the MCP connection and cleanup resources."""
        async with self.lock:
            await self._close_internal()
    
    async def _close_internal(self) -> None:
        """Internal close method without locking."""
        logger.info("Closing MCP connection")
        self.running = False
        
        # Cancel background tasks
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self._drain_task(self.heartbeat_task)
            finally:
                self.heartbeat_task = None
        
        if self.reconnect_task:
            self.reconnect_task.cancel()
            try:
                await self._drain_task(self.reconnect_task)
            finally:
                self.reconnect_task = None
        
        # Close client if it has cleanup methods
        if self.client:
            # MultiServerMCPClient doesn't have explicit close method
            # but we clear the reference
            self.client = None
        
        # Clear state
        self.server_config = None
        self.tools_cache = []
        self.last_heartbeat_ok = False
        self._loop = None

    async def _drain_task(self, task: asyncio.Task) -> None:
        """Await a cancelled task on its owning loop to prevent pending destruction warnings."""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        try:
            task_loop = task.get_loop()
        except Exception:
            task_loop = None

        # If we are on the same loop, await directly
        if current_loop is not None and task_loop is current_loop:
            try:
                await task
            except asyncio.CancelledError:
                pass
            return

        # If the task has a loop, schedule a drain coroutine on that loop
        if task_loop is not None and not task_loop.is_closed():
            try:
                fut = asyncio.run_coroutine_threadsafe(self._drain_task_on_loop(task), task_loop)
                try:
                    fut.result(timeout=2)
                except Exception:
                    pass
            except Exception:
                pass

    async def _drain_task_on_loop(self, task: asyncio.Task) -> None:
        try:
            await task
        except asyncio.CancelledError:
            pass
    
    async def _heartbeat_loop(self, interval_sec: int = 45) -> None:
        """
        Heartbeat loop to keep SSE connection alive.
        
        Args:
            interval_sec: Interval between heartbeats in seconds
        """
        logger.info(f"Starting heartbeat loop with {interval_sec}s interval")
        
        while self.running:
            try:
                await asyncio.sleep(interval_sec)
                
                if not self.running:
                    break
                
                # Send lightweight request to test connection
                await self.client.get_tools()
                self.last_heartbeat_ok = True
                logger.debug("Heartbeat successful")
                
            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}")
                self.last_heartbeat_ok = False
                if not self.running:
                    break
                self._ensure_reconnect()
    
    def _ensure_reconnect(self) -> None:
        """Ensure reconnection task is started if not already running."""
        if self.reconnect_task is None or self.reconnect_task.done():
            logger.info("Starting reconnection task")
            self.reconnect_task = asyncio.create_task(self._reconnect_loop())
    
    async def _reconnect_loop(self, base_delay: float = 0.5, max_delay: float = 30.0) -> None:
        """
        Reconnection loop with exponential backoff and jitter.
        
        Args:
            base_delay: Base delay for exponential backoff
            max_delay: Maximum delay between attempts
        """
        attempt = 0
        
        while self.running and not self.last_heartbeat_ok:
            try:
                # Calculate delay with exponential backoff and full jitter
                delay = min(max_delay, base_delay * (2 ** attempt))
                jittered_delay = random.uniform(0, delay)
                
                logger.info(f"Reconnection attempt {attempt + 1} in {jittered_delay:.2f}s")
                await asyncio.sleep(jittered_delay)
                
                if not self.running:
                    break
                
                # Try to recreate client and validate
                new_client = MultiServerMCPClient(self.server_config)
                await new_client.get_tools()
                
                # Success - swap in new client
                async with self.lock:
                    self.client = new_client
                    self.tools_cache = []  # Clear cache to force refresh
                    self.last_heartbeat_ok = True
                
                logger.info("Reconnection successful")
                break
                
            except Exception as e:
                logger.warning(f"Reconnection attempt {attempt + 1} failed: {e}")
                attempt += 1
        
        # Clear reconnect task reference
        self.reconnect_task = None
    
    @property
    def is_connected(self) -> bool:
        """Check if the manager is connected and healthy."""
        return self.running and self.client is not None and self.last_heartbeat_ok


def create_single_server_config(server_url: str, timeout: int = 600, sse_read_timeout: int = 900) -> Dict[str, Dict]:
    """
    Create a configuration for a single MCP server.
    
    Args:
        server_url: The URL of the MCP server
        timeout: Connection timeout in seconds
        sse_read_timeout: SSE read timeout in seconds
    
    Returns:
        Server configuration dictionary
    """
    return {
        "default_server": {
            "transport": "sse",
            "url": server_url,
            "headers": None,
            "timeout": timeout,
            "sse_read_timeout": sse_read_timeout
        }
    }


def create_multi_server_config(servers: Dict[str, str], timeout: int = 600, sse_read_timeout: int = 900) -> Dict[str, Dict]:
    """
    Create a configuration for multiple MCP servers.
    
    Args:
        servers: Dictionary mapping server names to URLs
        timeout: Connection timeout in seconds
        sse_read_timeout: SSE read timeout in seconds
    
    Returns:
        Server configuration dictionary
    """
    config = {}
    for name, url in servers.items():
        config[name] = {
            "transport": "sse",
            "url": url,
            "headers": None,
            "timeout": timeout,
            "sse_read_timeout": sse_read_timeout
        }
    return config


def validate_server_url(url: str) -> bool:
    """
    Validate if a server URL is properly formatted.
    
    Args:
        url: The URL to validate
    
    Returns:
        True if valid, False otherwise
    """
    if not url:
        return False
    
    # Basic URL validation
    if not (url.startswith("http://") or url.startswith("https://")):
        return False
    
    return True


def get_default_server_config() -> Dict[str, str]:
    """Get default server configuration values."""
    return {
        "default_url": "http://localhost:8000/sse",
        "default_timeout": "600",
        "default_sse_timeout": "900"
    } 