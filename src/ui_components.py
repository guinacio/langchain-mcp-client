"""
UI components and interface functions for the LangChain MCP Client.

This module contains all the user interface components including
sidebar, tabs, and various UI utility functions.
"""

import streamlit as st
import json
import datetime
import traceback
import aiohttp
from typing import Dict, List, Optional, Tuple

from .database import PersistentStorageManager
from .llm_providers import (
    get_available_providers, get_provider_models, get_default_model,
    requires_api_key, create_llm_model, supports_streaming
)
from .mcp_client import (
    create_single_server_config, create_multi_server_config, MCPConnectionManager
)
from .agent_manager import create_agent_with_tools
from .utils import run_async, reset_connection_state, format_error_message, model_supports_tools, create_download_data
from .llm_providers import is_openai_reasoning_model, supports_streaming_for_reasoning_model


async def test_ollama_connection(base_url: str = "http://localhost:11434") -> Tuple[bool, List[str]]:
    """Test if Ollama API is accessible and return available models"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{base_url}/api/tags",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    models = [model["name"] for model in result.get("models", [])]
                    return True, models
                else:
                    return False, []
    except Exception as e:
        return False, []


async def fetch_openai_models(api_key: str) -> Tuple[bool, List[str], str]:
    """Fetch available OpenAI models and filter to chat/reasoning models, excluding o1 series."""
    if not api_key:
        return False, [], "Missing API key"
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.openai.com/v1/models",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as response:
                if response.status != 200:
                    return False, [], f"HTTP {response.status}"
                data = await response.json()
                ids = [m.get("id", "") for m in data.get("data", []) if m.get("id")]
                filtered: List[str] = []
                for name in ids:
                    lower = name.lower()
                    # include common chat models and advanced reasoning (o3/o4), exclude non-chat classes
                    include = (
                        lower.startswith("gpt-") or lower.startswith("o3-") or lower.startswith("o4-") or lower == "gpt-4" or lower == "gpt-3.5-turbo"
                    )
                    exclude = (
                        lower.startswith("o1") or "o1-" in lower or
                        "embedding" in lower or "audio" in lower or "whisper" in lower or
                        "realtime" in lower or "tts" in lower or "speech" in lower or
                        "dall" in lower or "image" in lower
                    )
                    if include and not exclude:
                        filtered.append(name)
                # De-duplicate and sort
                filtered = sorted(list({m for m in filtered}))
                return True, filtered, ""
    except Exception as e:
        return False, [], str(e)


async def fetch_anthropic_models(api_key: str) -> Tuple[bool, List[str], str]:
    """Fetch available Anthropic models via their models endpoint."""
    if not api_key:
        return False, [], "Missing API key"
    try:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.anthropic.com/v1/models",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as response:
                if response.status != 200:
                    return False, [], f"HTTP {response.status}"
                data = await response.json()
                # Response shape: { "data": [ {"id": "claude-...", ...}, ... ] }
                ids = [m.get("id", "") for m in data.get("data", []) if m.get("id")]
                filtered = [m for m in ids if m and m.startswith("claude-")]
                filtered = sorted(list({m for m in filtered}))
                return True, filtered, ""
    except Exception as e:
        return False, [], str(e)


async def fetch_google_models(api_key: str) -> Tuple[bool, List[str], str]:
    """Fetch available Google Gemini models and map to model IDs."""
    if not api_key:
        return False, [], "Missing API key"
    try:
        params = {"key": api_key}
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                params=params,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as response:
                if response.status != 200:
                    return False, [], f"HTTP {response.status}"
                data = await response.json()
                models = data.get("models", [])
                names: List[str] = []
                for m in models:
                    full_name = m.get("name", "")  # e.g., "models/gemini-1.5-pro"
                    if not full_name:
                        continue
                    model_id = full_name.split("/")[-1]
                    lower = model_id.lower()
                    if not lower.startswith("gemini-"):
                        continue
                    # Prefer models that support generateContent (chat)
                    methods = m.get("supportedGenerationMethods", []) or []
                    if methods and not any("generate" in method for method in methods):
                        continue
                    names.append(model_id)
                names = sorted(list({n for n in names}))
                return True, names, ""
    except Exception as e:
        return False, [], str(e)


def render_sidebar():
    """Render the main application sidebar with all configuration options."""
    with st.sidebar:
        st.title("LangChain MCP Client")
        st.caption("Configure model, memory, and MCP connections.")

        with st.container(border=True):
            st.subheader("Model")
            llm_config = render_llm_configuration()
            render_streaming_configuration(llm_config)

        with st.container(border=True):
            memory_config = render_memory_configuration()

        with st.container(border=True):
            render_server_configuration(llm_config, memory_config)

        with st.container(border=True):
            render_available_tools()


def render_llm_configuration() -> Dict:
    """Render LLM provider configuration section."""
    llm_provider = st.selectbox(
        "Select LLM provider",
        options=get_available_providers(),
        index=0,
        on_change=reset_connection_state
    )
    
    # Store LLM provider in session state for Config tab
    st.session_state.llm_provider = llm_provider
    
    # Handle Ollama specially
    if llm_provider == "Ollama":
        return render_ollama_configuration()
    else:
        return render_standard_llm_configuration(llm_provider)


def render_ollama_configuration() -> Dict:
    """Render Ollama-specific configuration with dynamic model fetching."""
    # Ollama server URL configuration
    ollama_url = st.text_input(
        "Ollama server URL",
        value="http://localhost:11434",
        help="URL of your Ollama server (default: http://localhost:11434)"
    )
    
    # Connection test and model fetching
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Connect to Ollama", type="primary"):
            with st.spinner("Testing Ollama connection..."):
                success, models = run_async(test_ollama_connection(ollama_url))
                
                if success and models:
                    st.session_state.ollama_connected = True
                    st.session_state.ollama_models = models
                    st.session_state.ollama_url = ollama_url
                elif success and not models:
                    st.session_state.ollama_connected = True
                    st.session_state.ollama_models = []
                    st.session_state.ollama_url = ollama_url
                    st.warning("Connected but no models found. Make sure you have models installed.", icon=":material/warning:")
                    st.caption("Install models using: `ollama pull <model-name>`")
                else:
                    st.session_state.ollama_connected = False
                    st.session_state.ollama_models = []
                    st.error("Failed to connect to Ollama", icon=":material/error:")
                    st.caption("Make sure Ollama is running: `ollama serve`")
    
    with col2:
        # Show connection status
        if st.session_state.get('ollama_connected', False):
            st.badge("Connected", icon=":material/check_circle:", color="green")
        else:
            st.badge("Not connected", icon=":material/error:", color="red")
    
    # Model selection
    if st.session_state.get('ollama_connected', False):
        available_models = st.session_state.get('ollama_models', [])
        
        if available_models:
            model_count = len(st.session_state.get('ollama_models', []))
            # Show available models            
            st.write(f"**Available Models:** :blue-badge[{model_count} models available]")
            model_name = st.selectbox(
                "Select model",
                options=available_models,
                index=0,
                key="ollama_model_selector"
            )
            
        else:
            st.warning("No models found on Ollama server", icon=":material/warning:")
            st.caption("Install models using: `ollama pull <model-name>`")
            
            # Fallback to manual model input
            model_name = st.text_input(
                "Manual model name",
                placeholder="Enter model name (e.g. llama3.2, granite3.3:8b)",
                help="Enter the exact model name if you know it exists"
            )
            
            if not model_name:
                model_name = "llama3"  # Default fallback
    else:
        # Not connected - show manual input
        st.caption("Click 'Connect to Ollama' first to see available models")
        model_name = st.text_input(
            "Model name",
            value="granite3.3:8b",
            placeholder="Enter Ollama model name",
            help="Enter the model name. Connect to see available models."
        )
    
    # Store selected model in session state for Config tab
    st.session_state.selected_model = model_name
    
    return {
        "provider": "Ollama",
        "api_key": "",  # Ollama doesn't need API key
        "model": model_name,
        "ollama_url": ollama_url
    }


def render_standard_llm_configuration(llm_provider: str) -> Dict:
    """Render standard LLM configuration for non-Ollama providers."""
    # API Key input
    api_key = st.text_input(
        f"{llm_provider} API Key",
        type="password",
        help=f"Enter your {llm_provider} API Key"
    )
    
    # Store API key in session state for Config tab
    st.session_state.api_key = api_key
    
    # Dynamic model fetching for cloud providers
    dynamic_key = f"{llm_provider}_models_dynamic"
    fetched_flag_key = f"{llm_provider}_models_fetched"
    error_key = f"{llm_provider}_models_error"
    if fetched_flag_key not in st.session_state:
        st.session_state[fetched_flag_key] = False
    if dynamic_key not in st.session_state:
        st.session_state[dynamic_key] = []
    if error_key not in st.session_state:
        st.session_state[error_key] = ""

    if llm_provider in ("OpenAI", "Anthropic", "Google"):
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button(f"Fetch {llm_provider} models"):
                if not api_key:
                    st.session_state[error_key] = "Enter API key first"
                else:
                    with st.spinner("Fetching models..."):
                        if llm_provider == "OpenAI":
                            success, models, err = run_async(lambda: fetch_openai_models(api_key))
                        elif llm_provider == "Anthropic":
                            success, models, err = run_async(lambda: fetch_anthropic_models(api_key))
                        else:  # Google
                            success, models, err = run_async(lambda: fetch_google_models(api_key))
                        if success and models:
                            st.session_state[dynamic_key] = models
                            st.session_state[fetched_flag_key] = True
                            st.session_state[error_key] = ""
                        else:
                            st.session_state[dynamic_key] = []
                            st.session_state[fetched_flag_key] = False
                            st.session_state[error_key] = err or "No models returned"
        with col2:
            if st.session_state.get(fetched_flag_key):
                count = len(st.session_state.get(dynamic_key, []))
                st.badge(f"Fetched {count} models", icon=":material/check_circle:", color="green")
            else:
                st.badge("Not fetched", icon=":material/error:", color="red")
            if st.session_state.get(error_key):
                st.caption(f"Last error: {st.session_state.get(error_key)}")

    # Model selection (prefer dynamically fetched list if available)
    dynamic_models = st.session_state.get(dynamic_key, []) if llm_provider in ("OpenAI", "Anthropic", "Google") else []
    if dynamic_models:
        model_options = dynamic_models + ["Other"]
    else:
        model_options = get_provider_models(llm_provider)
    default_model = get_default_model(llm_provider)
    default_idx = model_options.index(default_model) if default_model in model_options else 0
    
    model_name = st.selectbox(
        "Model",
        options=model_options,
        index=default_idx,
        on_change=reset_connection_state
    )
    
    # Custom model input for providers that support "Other"
    if model_name == "Other":
        placeholder_text = {
            "OpenAI": "Enter custom OpenAI model name (e.g. gpt-4o, gpt-4, o3-mini)",
            "Anthropic": "Enter custom Anthropic model name (e.g. claude-3-sonnet-20240229)",
            "Google": "Enter custom Google model name (e.g. gemini-pro)"
        }.get(llm_provider, "Enter custom model name")
        
        custom_model = st.text_input(
            "Custom model name",
            placeholder=placeholder_text,
            key=f"custom_model_{llm_provider}"
        )
        if custom_model:
            model_name = custom_model
    
    # Show warning for reasoning models
    if llm_provider == "OpenAI" and is_openai_reasoning_model(model_name):
        # Check if it's an o1 series model (not supported)
        if model_name in ["o1", "o1-mini", "o1-preview"] or "o1-" in model_name.lower():
            st.error("**o1 series models are not supported**: o1, o1-mini, and o1-preview models have unique API requirements that are not compatible with this application. Please use o3-mini, o4-mini, or regular GPT models instead.", icon=":material/error:")
            st.info("**Recommended alternatives**: o3-mini, o4-mini, gpt-4o, or gpt-4 work great with this application.", icon=":material/tips_and_updates:")
        elif supports_streaming_for_reasoning_model(model_name):
            st.warning("**Reasoning model detected**: This model has special requirements - temperature is not supported. The model will use optimized parameters automatically.", icon=":material/warning:")
        else:
            st.warning("**Reasoning model detected**: This model has special requirements - temperature and streaming are not supported. The model will use optimized parameters automatically.", icon=":material/warning:")
    
    # Store selected model in session state for Config tab
    st.session_state.selected_model = model_name
    
    return {
        "provider": llm_provider,
        "api_key": api_key,
        "model": model_name
    }


def render_streaming_configuration(llm_config: Dict) -> None:
    """Render streaming configuration options."""
    with st.expander("Streaming settings", expanded=False, icon=":material/stream:"):
        provider = llm_config.get("provider", "")
        streaming_supported = supports_streaming(provider) if provider else False
        
        if streaming_supported:
            enable_streaming = st.checkbox(
                "Enable streaming",
                value=st.session_state.get('enable_streaming', True),
                help="Stream responses token by token for a more interactive experience"
            )
            st.session_state.enable_streaming = enable_streaming
            
            if enable_streaming:
                st.caption(":material/check_circle: Streaming enabled - responses will appear in real-time")
            else:
                st.caption(":material/info: Streaming disabled - responses will appear all at once")
        else:
            st.session_state.enable_streaming = False
            if provider:
                st.warning(f"{provider} does not support streaming", icon=":material/warning:")
            else:
                st.caption("Select a provider to see streaming options")


def render_memory_configuration() -> Dict:
    """Render memory configuration section."""
    st.subheader("Memory")
    
    memory_enabled = st.checkbox(
        "Enable conversation memory",
        value=st.session_state.get('memory_enabled', False),
        help="Enable persistent conversation memory across interactions",
        key="sidebar_memory_enabled"
    )
    st.session_state.memory_enabled = memory_enabled
    
    memory_config = {"enabled": memory_enabled}
    
    if memory_enabled:
        # Memory type selection
        memory_type = st.selectbox(
            "Memory type",
            options=["Short-term (Session)", "Persistent (Cross-session)"],
            index=0 if st.session_state.get('memory_type', 'Short-term (Session)') == 'Short-term (Session)' else 1,
            help="Short-term: Remembers within current session\nPersistent: Remembers across sessions using SQLite database",
        )
        st.session_state.memory_type = memory_type
        memory_config["type"] = memory_type
        
        # Initialize persistent storage if needed
        if memory_type == "Persistent (Cross-session)":
            if 'persistent_storage' not in st.session_state:
                st.session_state.persistent_storage = PersistentStorageManager()
            
            render_persistent_storage_section()
        
        # Thread ID management
        thread_id = st.text_input(
            "Conversation ID",
            value=st.session_state.get('thread_id', 'default'),
            help="Unique identifier for this conversation thread"
        )
        st.session_state.thread_id = thread_id
        memory_config["thread_id"] = thread_id
        
        # Memory management options
        render_memory_management_section()
    
    # Reset connection when memory settings change
    if st.session_state.get('_last_memory_enabled') != memory_enabled:
        reset_connection_state()
        st.session_state._last_memory_enabled = memory_enabled
    
    return memory_config


def render_persistent_storage_section():
    """Render persistent storage configuration and management."""
    with st.expander("Database settings", icon=":material/save:"):
        if hasattr(st.session_state, 'persistent_storage'):
            db_stats = st.session_state.persistent_storage.get_database_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Conversations", db_stats.get('conversation_count', 0))
                st.metric("Total messages", db_stats.get('total_messages', 0))
            with col2:
                st.metric("Database size", f"{db_stats.get('database_size_mb', 0)} MB")
                st.text(f"Path: {db_stats.get('database_path', 'N/A')}")
            
            # Conversation browser
            render_conversation_browser()


def render_conversation_browser():
    """Render conversation browser for persistent storage."""
    if not hasattr(st.session_state, 'persistent_storage'):
        return
    
    conversations = st.session_state.persistent_storage.list_conversations()
    if conversations:
        st.subheader("Saved conversations")
        for conv in conversations[:5]:  # Show last 5 conversations
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    display_title = conv.get('title') or conv['thread_id']
                    if len(display_title) > 30:
                        display_title = display_title[:30] + "..."
                    st.write(f"**{display_title}**")
                    last_msg = conv.get('last_message', '')
                    if last_msg and len(last_msg) > 50:
                        last_msg = last_msg[:50] + "..."
                    st.caption(f"{conv.get('message_count', 0)} messages • {last_msg}")
                
                with col2:
                    if st.button("Load", key=f"load_{conv['thread_id']}", icon=":material/folder_open:"):
                        st.session_state.thread_id = conv['thread_id']
                        st.session_state.chat_history = []
                        # Load conversation messages from database
                        if hasattr(st.session_state, 'persistent_storage'):
                            try:
                                loaded_messages = st.session_state.persistent_storage.load_conversation_messages(conv['thread_id'])
                                if loaded_messages:
                                    st.session_state.chat_history = loaded_messages
                            except Exception as e:
                                st.warning(f"Could not load conversation history: {str(e)}", icon=":material/warning:")
                        st.success(f"Loaded conversation: {conv['thread_id']}", icon=":material/check_circle:")
                        st.rerun()
                
                with col3:
                    if st.button("Delete", key=f"del_{conv['thread_id']}", icon=":material/delete:"):
                        if st.session_state.persistent_storage.delete_conversation(conv['thread_id']):
                            st.success("Conversation deleted", icon=":material/check_circle:")
                            st.rerun()


def render_memory_management_section():
    """Render memory management options."""
    with st.expander("Memory management"):
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Clear memory", icon=":material/delete:"):
                if hasattr(st.session_state, 'checkpointer') and st.session_state.checkpointer:
                    try:
                        st.session_state.chat_history = []
                        if hasattr(st.session_state, 'agent') and st.session_state.agent:
                            st.session_state.agent = None
                        st.success("Memory cleared successfully", icon=":material/check_circle:")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error clearing memory: {str(e)}", icon=":material/error:")
        
        with col2:
            max_messages = st.number_input(
                "Max messages",
                min_value=10,
                max_value=1000,
                value=st.session_state.get('max_messages', 100),
                help="Maximum messages to keep in memory"
            )
            st.session_state.max_messages = max_messages
        
        # Memory status
        if 'chat_history' in st.session_state:
            current_messages = len(st.session_state.chat_history)
            st.caption(f"Current conversation: {current_messages} messages")
        
        # Persistent storage actions
        render_persistent_storage_actions()


def render_persistent_storage_actions():
    """Render persistent storage action buttons."""
    memory_type = st.session_state.get('memory_type', '')
    if (memory_type == "Persistent (Cross-session)" and 
        hasattr(st.session_state, 'persistent_storage')):
        
        st.subheader("Persistent storage actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save current conversation", icon=":material/save:"):
                if st.session_state.chat_history:
                    # Generate a title from the first user message
                    title = None
                    for msg in st.session_state.chat_history[:3]:
                        if msg.get('role') == 'user':
                            title = msg.get('content', '')[:50] + "..." if len(msg.get('content', '')) > 50 else msg.get('content', '')
                            break
                    
                    thread_id = st.session_state.get('thread_id', 'default')
                    st.session_state.persistent_storage.update_conversation_metadata(
                        thread_id=thread_id,
                        title=title,
                        message_count=len(st.session_state.chat_history),
                        last_message=st.session_state.chat_history[-1].get('content', '') if st.session_state.chat_history else ''
                    )
                    st.success("Conversation metadata saved", icon=":material/check_circle:")
                else:
                    st.warning("No conversation to save", icon=":material/warning:")
        
        with col2:
            if st.button("Export conversation", icon=":material/upload:"):
                thread_id = st.session_state.get('thread_id', 'default')
                export_data = st.session_state.persistent_storage.export_conversation(thread_id)
                if export_data:
                    json_str, filename = create_download_data(export_data, f"conversation_{thread_id}")
                    st.download_button(
                        label="Download export",
                        data=json_str,
                        file_name=filename,
                        mime="application/json",
                        icon=":material/download:",
                    )
                else:
                    st.error("Failed to export conversation", icon=":material/error:")


def render_server_configuration(llm_config: Dict, memory_config: Dict) -> Dict:
    """Render MCP server configuration section."""
    st.subheader("MCP servers")
    
    provider = llm_config.get("provider", "")
    default_index = 0  # Index 0 = "Single Server"
    
    server_mode = st.radio(
        "Server mode",
        options=["Single Server", "Multiple Servers", "No MCP Server (Chat Only)"],
        index=default_index,
        key=f"server_mode_radio_{provider}",  # Force re-render when provider changes
        on_change=lambda: setattr(st.session_state, "current_tab", server_mode)
    )
    
    if server_mode == "Single Server":
        return render_single_server_config(llm_config, memory_config)
    elif server_mode == "Multiple Servers":
        return render_multiple_servers_config(llm_config, memory_config)
    else:  # Chat-only mode
        return render_chat_only_config(llm_config, memory_config)


def render_single_server_config(llm_config: Dict, memory_config: Dict) -> Dict:
    """Render single server configuration."""
    server_url = st.text_input(
        "MCP server URL",
        value="http://localhost:8000/sse",
        help="Enter the URL of your MCP server (SSE endpoint)"
    )
    
    if st.button("Connect to MCP Server", type="primary"):
        return handle_single_server_connection(llm_config, memory_config, server_url)
    
    return {"mode": "single", "connected": False}


def render_multiple_servers_config(llm_config: Dict, memory_config: Dict) -> Dict:
    """Render multiple servers configuration."""
    st.subheader("Server management")
    
    # Server input
    server_name = st.text_input(
        "Server name",
        value="",
        help="Enter a unique name for this server (e.g., 'weather', 'math')"
    )
    
    server_url = st.text_input(
        "Server URL",
        value="http://localhost:8000/sse",
        help="Enter the URL of the MCP server (SSE endpoint)"
    )
    
    # Add server button
    if st.button("Add Server"):
        handle_add_server(server_name, server_url)
    
    # Display configured servers
    render_configured_servers()
    
    # Connect to all servers
    if st.button("Connect to All Servers"):
        return handle_multiple_servers_connection(llm_config, memory_config)
    
    return {"mode": "multiple", "connected": False}


def render_chat_only_config(llm_config: Dict, memory_config: Dict) -> Dict:
    """Render chat-only mode configuration."""
    st.subheader("Direct chat mode")
    st.info("This mode provides a direct chat interface with the LLM without any MCP tools.", icon=":material/chat_bubble:")
    
    if st.button("Start Chat Agent", type="primary"):
        return handle_chat_only_connection(llm_config, memory_config)
    
    # Information about chat-only mode
    with st.expander("About chat-only mode", icon=":material/info:"):
        st.markdown("""
        **Chat-Only Mode** provides a direct interface to the selected LLM without any MCP server tools.
        
        **Features:**
        - Direct conversation with the LLM
        - Memory support (if enabled)
        - No external tool dependencies
        - Faster setup and response times
        
        **Use Cases:**
        - General conversation and Q&A
        - Creative writing and brainstorming
        - Learning and explanations
        - Code review and discussion (without execution)
        
        **Note:** In this mode, the agent cannot perform external actions like web searches, file operations, or API calls that would normally be available through MCP tools.
        """)
    
    return {"mode": "chat_only", "connected": False}


def handle_single_server_connection(llm_config: Dict, memory_config: Dict, server_url: str) -> Dict:
    """Handle single server connection logic with improved error handling."""
    # Check API key requirement for non-Ollama providers
    if (llm_config["provider"] != "Ollama" and 
        not llm_config["api_key"] and 
        requires_api_key(llm_config["provider"])):
        st.error(f"Please enter your {llm_config['provider']} API Key")
        return {"mode": "single", "connected": False}
    
    # Check Ollama connection if it's the selected provider
    if (llm_config["provider"] == "Ollama" and 
        not st.session_state.get('ollama_connected', False)):
        st.error("Please test Ollama connection first")
        return {"mode": "single", "connected": False}
    elif not server_url:
        st.error("Please enter a valid MCP Server URL")
        return {"mode": "single", "connected": False}
    
    # Create progress container
    progress_container = st.container()
    
    with progress_container:
        with st.spinner("Connecting to MCP server..."):
            try:
                # Show progress steps
                progress_placeholder = st.empty()
                
                # Step 1: Setup configuration
                progress_placeholder.info("Setting up server configuration...", icon=":material/handyman:")
                server_config = create_single_server_config(
                    server_url, 
                    timeout=60,  # 1 minute for initial connection
                    sse_read_timeout=300  # 5 minutes for SSE operations
                )
                
                # Step 2: Initialize MCP connection manager
                progress_placeholder.info("Initializing MCP connection manager...", icon=":material/language:")
                
                # Get or create the connection manager
                mcp_manager = MCPConnectionManager.get_instance()
                st.session_state.mcp_manager = mcp_manager
                
                # Start the connection
                try:
                    run_async(lambda: mcp_manager.start(server_config))
                    if not mcp_manager.is_connected:
                        progress_placeholder.error("Failed to start MCP connection manager", icon=":material/error:")
                        return {"mode": "single", "connected": False}
                except Exception as e:
                    formatted_error = format_error_message(e)
                    progress_placeholder.error(f"Failed to start MCP connection manager: {formatted_error}", icon=":material/error:")
                    return {"mode": "single", "connected": False}
                
                # Step 3: Get tools
                progress_placeholder.info("Retrieving tools from server...", icon=":material/search:")
                try:
                    tools = run_async(lambda: mcp_manager.get_tools())
                    if tools is None:
                        tools = []
                except Exception as e:
                    formatted_error = format_error_message(e)
                    progress_placeholder.error(f"Failed to retrieve tools: {formatted_error}", icon=":material/error:")
                    return {"mode": "single", "connected": False}
                
                st.session_state.tools = tools
                
                # Step 4: Create agent
                progress_placeholder.info("Creating and configuring agent...", icon=":material/smart_toy:")
                success = create_and_configure_agent(llm_config, memory_config, st.session_state.tools)
                
                if success:
                    progress_placeholder.empty()  # Clear progress messages
                    st.success(f"Connected to MCP server. Found {len(st.session_state.tools)} tools.", icon=":material/check_circle:")
                    with st.expander("Connection details", icon=":material/handyman:"):
                        st.write(f"**Server URL:** {server_url}")
                        st.write(f"**Tools found:** {len(st.session_state.tools)}")
                        st.write(f"**Connection timeout:** 1 minute")
                        st.write(f"**SSE read timeout:** 5 minutes")
                    return {"mode": "single", "connected": True}
                else:
                    progress_placeholder.error("Failed to configure agent", icon=":material/error:")
                    return {"mode": "single", "connected": False}
                    
            except Exception as e:
                formatted_error = format_error_message(e)
                st.error(f"Error connecting to MCP server: {formatted_error}", icon=":material/error:")
                
                # Show additional troubleshooting info
                with st.expander("Troubleshooting", icon=":material/search:"):
                    st.write("**Common solutions:**")
                    st.write("• Check that the MCP server is running and accessible")
                    st.write("• Verify the server URL is correct")
                    st.write("• Try refreshing the page and reconnecting")
                    st.write("• For external servers, ensure they support SSE connections")
                    st.write("• Check if the server has rate limiting or requires authentication")
                    
                    st.write("**Technical details:**")
                    st.code(traceback.format_exc(), language="python")
                
                return {"mode": "single", "connected": False}


def handle_multiple_servers_connection(llm_config: Dict, memory_config: Dict) -> Dict:
    """Handle multiple servers connection logic with improved error handling."""
    # Check API key requirement for non-Ollama providers
    if (llm_config["provider"] != "Ollama" and 
        not llm_config["api_key"] and 
        requires_api_key(llm_config["provider"])):
        st.error(f"Please enter your {llm_config['provider']} API Key")
        return {"mode": "multiple", "connected": False}
    
    # Check Ollama connection if it's the selected provider
    if (llm_config["provider"] == "Ollama" and 
        not st.session_state.get('ollama_connected', False)):
        st.error("Please test Ollama connection first")
        return {"mode": "multiple", "connected": False}
    elif not st.session_state.servers:
        st.error("Please add at least one server")
        return {"mode": "multiple", "connected": False}
    
    with st.spinner("Connecting to MCP servers..."):
        try:
            # Initialize the MCP connection manager with all servers
            mcp_manager = MCPConnectionManager.get_instance()
            st.session_state.mcp_manager = mcp_manager
            
            # Start the connection with multiple servers config
            try:
                run_async(lambda: mcp_manager.start(st.session_state.servers))
                if not mcp_manager.is_connected:
                    return {"mode": "multiple", "connected": False}
            except Exception as e:
                formatted_error = format_error_message(e)
                st.error(f"Failed to start MCP connection manager: {formatted_error}", icon=":material/error:")
                return {"mode": "multiple", "connected": False}
            
            # Get tools from the manager
            try:
                tools = run_async(lambda: mcp_manager.get_tools())
                if tools is None:
                    tools = []
            except Exception as e:
                formatted_error = format_error_message(e)
                st.error(f"Failed to retrieve tools: {formatted_error}", icon=":material/error:")
                return {"mode": "multiple", "connected": False}
            
            st.session_state.tools = tools
            
            # Create and configure agent
            success = create_and_configure_agent(llm_config, memory_config, st.session_state.tools)
            
            if success:
                st.success(f"Connected to {len(st.session_state.servers)} MCP servers. Found {len(st.session_state.tools)} tools.", icon=":material/check_circle:")
                with st.expander("Connection details", icon=":material/handyman:"):
                    st.write(f"**Servers connected:** {len(st.session_state.servers)}")
                    for name, config in st.session_state.servers.items():
                        st.write(f"  • {name}: {config['url']}")
                    st.write(f"**Total tools found:** {len(st.session_state.tools)}")
                return {"mode": "multiple", "connected": True}
            else:
                return {"mode": "multiple", "connected": False}
                
        except Exception as e:
            formatted_error = format_error_message(e)
            st.error(f"Error connecting to MCP servers: {formatted_error}", icon=":material/error:")
            
            # Show additional troubleshooting info
            with st.expander("Troubleshooting", icon=":material/search:"):
                st.write("**Common solutions:**")
                st.write("• Check that all MCP servers are running and accessible")
                st.write("• Verify all server URLs are correct")
                st.write("• Try connecting to servers individually first")
                st.write("• For external servers, ensure they support SSE connections")
                
                st.write("**Configured servers:**")
                for name, config in st.session_state.servers.items():
                    st.write(f"  • {name}: {config['url']}")
                
                st.write("**Technical details:**")
                st.code(traceback.format_exc(), language="python")
            
            return {"mode": "multiple", "connected": False}


def handle_chat_only_connection(llm_config: Dict, memory_config: Dict) -> Dict:
    """Handle chat-only mode connection logic."""
    # Check API key requirement for non-Ollama providers
    if (llm_config["provider"] != "Ollama" and 
        not llm_config["api_key"] and 
        requires_api_key(llm_config["provider"])):
        st.error(f"Please enter your {llm_config['provider']} API Key")
        return {"mode": "chat_only", "connected": False}
    
    # Check Ollama connection if it's the selected provider
    if (llm_config["provider"] == "Ollama" and 
        not st.session_state.get('ollama_connected', False)):
        st.error("Please test Ollama connection first")
        return {"mode": "chat_only", "connected": False}
    
    with st.spinner("Initializing chat agent..."):
        try:
            # Clear any existing MCP tools since we're in chat-only mode
            st.session_state.tools = []
            # Close any existing MCP manager since we're in chat-only mode
            if st.session_state.get('mcp_manager'):
                try:
                    run_async(lambda: st.session_state.mcp_manager.close())
                except Exception:
                    pass
                st.session_state.mcp_manager = None
            
            # Create and configure agent
            success = create_and_configure_agent(llm_config, memory_config, [])
            
            if success:
                # Check if the model supports tools
                model_name = llm_config.get('model', '')
                supports_tools = model_supports_tools(model_name)
                
                # Determine appropriate success message
                if not supports_tools:
                    if memory_config.get("enabled"):
                        st.success("Chat agent ready (memory enabled, but model does not support tools).", icon=":material/check_circle:")
                        st.info("This model does not support tool calling, so memory will work through conversation history only.", icon=":material/info:")
                    else:
                        st.success("Chat agent ready (simple chat mode - no tools, no memory).", icon=":material/check_circle:")
                else:
                    if memory_config.get("enabled"):
                        st.success("Chat agent ready (memory enabled with history tool).", icon=":material/check_circle:")
                    else:
                        st.success("Chat agent ready (no additional tools).", icon=":material/check_circle:")
                return {"mode": "chat_only", "connected": True}
            else:
                return {"mode": "chat_only", "connected": False}
                
        except Exception as e:
            error_message = str(e)
            if "does not support tools" in error_message:
                st.error("This model does not support tool calling. The agent has been configured in simple chat mode.", icon=":material/error:")
                st.info("ℹ️ Memory and tool features are disabled for this model, but basic conversation works.")
            else:
                st.error(f"Error initializing chat agent: {error_message}")
            st.code(traceback.format_exc(), language="python")
            return {"mode": "chat_only", "connected": False}


def create_and_configure_agent(llm_config: Dict, memory_config: Dict, mcp_tools: List) -> bool:
    """Create and configure the agent with the given parameters."""
    try:
        # Get configuration parameters from session state
        use_custom_config = st.session_state.get('config_use_custom_settings', False)
        
        if use_custom_config:
            # Use custom configuration parameters
            temperature = st.session_state.get('config_temperature', 0.7)
            max_tokens = st.session_state.get('config_max_tokens')
            timeout = st.session_state.get('config_timeout')
            system_prompt = st.session_state.get('config_system_prompt')
        else:
            # Use default parameters
            temperature = 0.7
            max_tokens = None
            timeout = None
            system_prompt = None
        
        # Create the language model with configuration
        llm = create_llm_model(
            llm_provider=llm_config["provider"], 
            api_key=llm_config["api_key"], 
            model_name=llm_config["model"],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            system_prompt=system_prompt,
            ollama_url=llm_config.get("ollama_url")
        )
        
        # Get persistent storage if needed
        persistent_storage = None
        if (memory_config.get("enabled") and 
            memory_config.get("type") == "Persistent (Cross-session)" and
            hasattr(st.session_state, 'persistent_storage')):
            persistent_storage = st.session_state.persistent_storage
        
        # Create the agent
        agent, checkpointer = create_agent_with_tools(
            llm=llm,
            mcp_tools=mcp_tools,
            memory_enabled=memory_config.get("enabled", False),
            memory_type=memory_config.get("type", "Short-term (Session)"),
            persistent_storage=persistent_storage
        )
        
        st.session_state.agent = agent
        st.session_state.checkpointer = checkpointer
        
        return True
        
    except Exception as e:
        st.error(f"Error creating agent: {str(e)}")
        return False


def handle_add_server(server_name: str, server_url: str):
    """Handle adding a new server to the configuration."""
    if not server_name:
        st.error("Please enter a server name")
    elif not server_url:
        st.error("Please enter a valid server URL")
    elif server_name in st.session_state.servers:
        st.error(f"Server '{server_name}' already exists")
    else:
        st.session_state.servers[server_name] = {
            "transport": "sse",
            "url": server_url,
            "headers": None,
            "timeout": 600,
            "sse_read_timeout": 900
        }
        st.success(f"Added server '{server_name}'")


def render_configured_servers():
    """Render the list of configured servers."""
    if st.session_state.servers:
        st.subheader("Configured servers")
        for name, config in st.session_state.servers.items():
            with st.expander(f"Server: {name}"):
                st.write(f"**URL:** {config['url']}")
                if st.button(f"Remove {name}", key=f"remove_{name}"):
                    del st.session_state.servers[name]
                    st.rerun()


def render_available_tools():
    """Render the available tools section with connection status."""
    # Show connection status
    mcp_manager = st.session_state.get('mcp_manager')
    if mcp_manager:
        if mcp_manager.is_connected:
            st.success("MCP connection: Connected", icon=":material/check_circle:")
        elif mcp_manager.running:
            st.warning("MCP connection: Reconnecting...", icon=":material/pending:")
        else:
            st.error("MCP connection: Disconnected", icon=":material/error:")
    else:
        st.caption("MCP connection: :orange-badge[Not initialized]")
    
    # Always show the tools header if we have a manager
    if mcp_manager or st.session_state.get('agent'):
        st.subheader("Available tools")
        
        # Add refresh button for tools
        if mcp_manager and st.button("Refresh tools", icon=":material/sync:"):
            try:
                tools = run_async(lambda: mcp_manager.get_tools(force_refresh=True))
                st.session_state.tools = tools or []
                st.rerun()
            except Exception as e:
                st.error(f"Failed to refresh tools: {format_error_message(e)}")
        
        # Check if the current model supports tools
        model_name = st.session_state.get('selected_model', '')
        supports_tools = model_supports_tools(model_name)
        
        # Show total tool count including history tool
        mcp_tool_count = len(st.session_state.tools)
        memory_tool_count = 1 if st.session_state.get('memory_enabled', False) and supports_tools else 0
        total_tools = mcp_tool_count + memory_tool_count
    
    if st.session_state.tools or (st.session_state.agent and st.session_state.get('memory_enabled', False)):
        
        if not supports_tools and st.session_state.get('memory_enabled', False):
            st.info("Memory enabled (conversation history only - model does not support tool calling)", icon=":material/psychology:")
            st.warning("This model does not support tools, so the history tool is not available. Memory works through conversation context only.", icon=":material/warning:")
        elif mcp_tool_count > 0 and memory_tool_count > 0:
            st.caption(f"{total_tools} tools available ({mcp_tool_count} MCP + {memory_tool_count} memory tool)")
        elif mcp_tool_count > 0:
            st.caption(f"{mcp_tool_count} MCP tools available")
        elif memory_tool_count > 0:
            st.caption(f"{memory_tool_count} memory tool available")
        else:
            st.caption("No tools available (chat-only mode)")
        
        render_tool_selector()
    elif st.session_state.agent:
        # Agent exists but no tools - show appropriate message
        model_name = st.session_state.get('selected_model', '')
        if not model_supports_tools(model_name):
            st.subheader("Agent status")
            st.caption("Simple chat mode - this model does not support tool calling")
        else:
            st.subheader("Available tools")
            st.caption("No tools available (chat-only mode)")


def render_tool_selector():
    """Render the tool selection dropdown and information."""
    # Check if the current model supports tools
    model_name = st.session_state.get('selected_model', '')
    supports_tools = model_supports_tools(model_name)
    
    # Add history tool to the dropdown when memory is enabled AND model supports tools
    tool_options = [tool.name for tool in st.session_state.tools]
    if st.session_state.get('memory_enabled', False) and supports_tools:
        tool_options.append("get_conversation_history (Memory)")
    
    # Only show tool selection if there are tools available
    if tool_options:
        selected_tool_name = st.selectbox(
            "Available tools",
            options=tool_options,
            index=0 if tool_options else None
        )
        
        if selected_tool_name:
            render_tool_information(selected_tool_name)
    else:
        # No tools available
        if st.session_state.get('memory_enabled', False) and not supports_tools:
            st.caption("Memory is enabled but works through conversation context only (no tool interface)")
        else:
            st.caption("In chat-only mode - no external tools available")


def render_tool_information(selected_tool_name: str):
    """Render detailed information about the selected tool."""
    if selected_tool_name == "get_conversation_history (Memory)":
        st.write("**Description:** Retrieve conversation history from the current session with advanced filtering and search options")
        st.write("**Enhanced Features:**")
        st.write("• Timestamps and message IDs for precise referencing")
        st.write("• Date range filtering and flexible sorting")
        st.write("• Rich metadata including tool execution details")
        st.write("• Advanced search with boolean operators and regex support")
        
        st.write("**Parameters:**")
        st.code("message_type: string (optional) [default: all]")
        st.code("last_n_messages: integer (optional) [default: 10, max: 100]") 
        st.code("search_query: string (optional) - supports text, boolean ops, regex")
        st.code("sort_order: string (optional) [default: newest_first]")
        st.code("date_from: string (optional) [YYYY-MM-DD format]")
        st.code("date_to: string (optional) [YYYY-MM-DD format]")
        st.code("include_metadata: boolean (optional) [default: true]")
        
        with st.expander("Advanced search examples", icon=":material/search:"):
            st.write("**Simple Text Search:**")
            st.code('search_query="weather"')
            
            st.write("**Boolean Operators:**")
            st.code('search_query="weather AND temperature"')
            st.code('search_query="sunny OR cloudy OR rainy"')
            st.code('search_query="weather NOT rain"')
            st.code('search_query="(weather OR climate) AND NOT error"')
            
            st.write("**Regex Patterns:**")
            st.code('search_query="regex:\\\\d{2}°[CF]"  # Find temperatures like "72°F"')
            st.code('search_query="regex:https?://\\\\S+"  # Find URLs')
            st.code('search_query="regex:\\\\$\\\\d+(\\\\.\\\\d{2})?"  # Find dollar amounts')
            st.code('search_query="regex:\\\\b\\\\d{4}-\\\\d{2}-\\\\d{2}\\\\b"  # Find dates')
            
            st.write("**Complex Queries:**")
            st.code('search_query="tool AND (success OR complete) NOT error"')
            st.code('search_query="regex:API.*key AND NOT expired"')
        
        st.info("This enhanced tool provides enterprise-grade conversation history access with powerful search capabilities including boolean logic and regex pattern matching.", icon=":material/tips_and_updates:")
    else:
        # Find the selected MCP tool
        selected_tool = next((tool for tool in st.session_state.tools if tool.name == selected_tool_name), None)
        
        if selected_tool:
            # Display tool information
            st.write(f"**Description:** {selected_tool.description}")
            
            # Display parameters if available
            if hasattr(selected_tool, 'args_schema'):
                st.write("**Parameters:**")
                render_tool_parameters(selected_tool)


def render_tool_parameters(tool):
    """Render tool parameters information."""
    # Get schema properties directly from the tool
    schema = getattr(tool, 'args_schema', {})
    if isinstance(schema, dict):
        properties = schema.get('properties', {})
        required = schema.get('required', [])
    else:
        # Handle Pydantic schema
        schema_dict = schema.schema()
        properties = schema_dict.get('properties', {})
        required = schema_dict.get('required', [])

    # Display each parameter with its details
    for param_name, param_info in properties.items():
        # Get parameter details
        param_type = param_info.get('type', 'string')
        param_title = param_info.get('title', param_name)
        param_default = param_info.get('default', None)
        is_required = param_name in required

        # Build parameter description
        param_desc = [
            f"{param_title}:",
            f"{param_type}",
            "(required)" if is_required else "(optional)"
        ]
        
        if param_default is not None:
            param_desc.append(f"[default: {param_default}]")

        # Display parameter info
        st.code(" ".join(param_desc), wrap_lines=True) 