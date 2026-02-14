"""
LangChain MCP Client - Main Application

A Streamlit application for interacting with LangChain MCP (Model Context Protocol) servers.
This refactored version uses modular components for better maintainability.
"""

import streamlit as st
import nest_asyncio
from pathlib import Path

# Apply nest_asyncio to allow nested asyncio event loops (needed for Streamlit's execution model)
nest_asyncio.apply()

# Import all our modular components
from src.utils import initialize_session_state
from src.ui_components import render_sidebar
from src.tab_components import (
    render_chat_tab,
    render_test_tools_tab, 
    render_memory_tab,
    render_config_tab,
    render_about_tab,
    display_tool_executions
)


def main():
    """Main application entry point."""
    # Set page configuration
    base_dir = Path(__file__).parent
    st.set_page_config(
        page_title="LangChain MCP Client",
        page_icon=str(base_dir / "logo_transparent.png"),
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Set logo
    st.logo(str(base_dir / "side_logo.png"), size="large")

    # Initialize session state with all required variables
    initialize_session_state()

    # Render sidebar with all configuration options
    render_sidebar()

    # Create main tabs
    tab_chat, tab_test, tab_memory, tab_config, tab_about = st.tabs([
        ":material/chat: Chat",
        ":material/handyman: Test tools",
        ":material/psychology: Memory",
        ":material/settings: Config",
        ":material/info: About",
    ])

    # Render tab content
    with tab_chat:
        render_chat_tab()
        display_tool_executions()
    
    with tab_test:
        render_test_tools_tab()
    
    with tab_memory:
        render_memory_tab()
    
    with tab_config:
        render_config_tab()
    
    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main() 