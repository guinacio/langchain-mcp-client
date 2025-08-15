"""
Tab components for the LangChain MCP Client.

This module contains all the tab rendering functions including
chat, tool testing, memory management, and about sections.
"""

import streamlit as st
import json
import datetime
import time
import traceback
import re
from typing import Dict, List, Any
from langchain_core.messages import HumanMessage

from .agent_manager import (
    run_agent, run_tool, prepare_agent_invocation_config,
    extract_tool_executions_from_response, extract_assistant_response,
    stream_agent_response, stream_agent_events
)
from .memory_tools import calculate_chat_statistics, format_chat_history_for_export
from .utils import (
    run_async, create_download_data, safe_async_call, format_error_message,
    run_streaming_async, coerce_content_to_text, build_multimodal_human_content,
    is_vision_model
)
from .database import PersistentStorageManager
from .llm_providers import (
    get_available_providers, supports_system_prompt, get_default_temperature,
    get_temperature_range, get_default_max_tokens, get_max_tokens_range,
    get_default_timeout, validate_model_parameters, DEFAULT_SYSTEM_PROMPT
)


def parse_reasoning_content(content: str) -> Dict[str, str]:
    """
    Parse content to extract thinking and final response.
    
    Args:
        content: The full content string that may contain <think></think> tags
        
    Returns:
        Dict with 'thinking' and 'response' keys
    """
    thinking_pattern = r'<think>(.*?)</think>'
    
    # Find all thinking sections
    thinking_matches = re.findall(thinking_pattern, content, re.DOTALL)
    
    # Remove all thinking sections to get the final response
    response_content = re.sub(thinking_pattern, '', content, flags=re.DOTALL).strip()
    
    # Combine all thinking sections
    thinking_content = '\n\n'.join(thinking_matches) if thinking_matches else ''
    
    return {
        'thinking': thinking_content,
        'response': response_content
    }


def detect_reasoning_in_stream(text_buffer: str, thinking_round: int = 1) -> Dict[str, Any]:
    """
    Detect reasoning tags in streaming text and return parsing information.
    
    Args:
        text_buffer: Current accumulated text buffer
        thinking_round: Which round of thinking we're looking for
        
    Returns:
        Dict with detection status and extracted content
    """
    result = {
        'has_thinking': False,
        'thinking_complete': False,
        'thinking_content': '',
        'response_content': '',
        'in_thinking': False
    }
    
    # Find all thinking blocks
    import re
    thinking_pattern = r'<think>(.*?)</think>'
    thinking_matches = list(re.finditer(thinking_pattern, text_buffer, re.DOTALL))
    
    # Check if we have the thinking round we're looking for
    if len(thinking_matches) >= thinking_round:
        # We have a complete thinking block for this round
        match = thinking_matches[thinking_round - 1]  # 0-indexed
        result['has_thinking'] = True
        result['thinking_complete'] = True
        result['thinking_content'] = match.group(1).strip()
        result['in_thinking'] = False
        
        # Extract response content (everything after the last complete thinking block)
        last_match = thinking_matches[-1]
        response_start = last_match.end()
        result['response_content'] = text_buffer[response_start:].strip()
        
    elif '<think>' in text_buffer:
        # We have an incomplete thinking block
        # Count complete blocks to see if this is a new round
        complete_blocks = len(thinking_matches)
        
        # Find the last opening tag
        last_think_start = text_buffer.rfind('<think>')
        
        if complete_blocks + 1 == thinking_round:
            # This is the thinking round we're looking for, but it's incomplete
            result['has_thinking'] = True
            result['in_thinking'] = True
            think_content_start = last_think_start + 7  # Length of '<think>'
            result['thinking_content'] = text_buffer[think_content_start:].strip()
    
    return result


def render_chat_tab():
    """Render the main chat interface tab."""
    # Connection and memory status indicators
    render_status_indicators()

    # Display chat history
    render_chat_history()

    # Chat input and processing
    handle_chat_input()


def render_status_indicators():
    """Render enhanced connection and memory status indicators."""
    # Create a more sophisticated status display
    with st.container():
        # Main status row
        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
        
        with col1:
            # Connection status with detailed info
            if st.session_state.agent is not None:
                if st.session_state.client is not None:
                    # Connected to MCP server(s)
                    server_count = len(st.session_state.get('servers', {})) if st.session_state.get('servers') else 1
                    tool_count = len(st.session_state.get('tools', []))
                    
                    with st.status("📶 MCP Connection Active", expanded=False, state="complete"):
                        if server_count > 1:
                            st.write(f"**Connected to {server_count} MCP servers**")
                        else:
                            st.write("**Connected to MCP server**")
                        
                        if tool_count > 0:
                            st.write(f"🔧 **{tool_count} tools available**")
                            
                            # Show tool categories if we have tools
                            tools = st.session_state.get('tools', [])
                            if tools:
                                tool_names = [tool.name for tool in tools[:5]]  # Show first 5
                                st.write("**Available tools:**")
                                for tool_name in tool_names:
                                    st.write(f"• {tool_name}")
                                if len(tools) > 5:
                                    st.write(f"• ... and {len(tools) - 5} more")
                        else:
                            st.write("⚠️ No tools available")
                else:
                    # Chat-only mode
                    with st.status("💬 Chat-Only Mode", expanded=False, state="complete"):
                        st.write("**Direct LLM conversation**")
                        st.write("No MCP server connected")
                        
                        # Show model info if available
                        provider = st.session_state.get('llm_provider', 'Unknown')
                        model = st.session_state.get('selected_model', 'Unknown')
                        st.write(f"**Model:** {provider} - {model}")
            else:
                # No agent initialized
                with st.status("⚠️ Agent Not Ready", expanded=False, state="error"):
                    st.write("**Please initialize an agent:**")
                    st.write("• Connect to MCP server, or")
                    st.write("• Start chat-only mode")
                    st.write("Use the sidebar to get started!")
        
        with col2:
            # Memory status with detailed info
            memory_enabled = st.session_state.get('memory_enabled', False)
            
            if memory_enabled:
                memory_type = st.session_state.get('memory_type', 'Short-term (Session)')
                thread_id = st.session_state.get('thread_id', 'default')
                chat_length = len(st.session_state.get('chat_history', []))
                max_messages = st.session_state.get('max_messages', 100)
                
                memory_status_label = "🧠 Memory Active"
                if memory_type == "Persistent (Cross-session)":
                    memory_status_label += " (Persistent)"
                
                with st.status(memory_status_label, expanded=False, state="complete"):
                    st.write(f"**Type:** {memory_type.split()[0]}")
                    st.write(f"**Thread:** {thread_id}")
                    st.write(f"**Messages:** {chat_length}/{max_messages}")
                    
                    # Memory usage indicator
                    usage_percent = (chat_length / max_messages) * 100
                    if usage_percent > 80:
                        st.warning(f"Memory usage: {usage_percent:.1f}% (Consider clearing)")
                    elif usage_percent > 50:
                        st.info(f"Memory usage: {usage_percent:.1f}%")
                    else:
                        st.success(f"Memory usage: {usage_percent:.1f}%")
                    
                    # Show persistent storage info if applicable
                    if (memory_type == "Persistent (Cross-session)" and 
                        hasattr(st.session_state, 'persistent_storage')):
                        db_stats = st.session_state.persistent_storage.get_database_stats()
                        st.write(f"**Database:** {db_stats.get('conversation_count', 0)} conversations")
            else:
                with st.status("🧠 Memory Disabled", expanded=False, state="running"):
                    st.write("**No conversation memory**")
                    st.write("Each message is independent")
                    st.info("Enable memory in sidebar for context retention")
        
        with col3:
            # Quick actions and settings
            with st.container():
                # Streaming status
                streaming_enabled = st.session_state.get('enable_streaming', True)
                if streaming_enabled:
                    st.success("🌊 Streaming ON")
                else:
                    st.info("🌊 Streaming OFF")
        with col4:
            with st.container():    # Configuration status
                custom_config = st.session_state.get('config_use_custom_settings', False)
                if custom_config:
                    st.info("⚙️ Custom Config")
                else:
                    st.text("⚙️ Default Config")
    
    # Show recent activity if available
    recent_tools = st.session_state.get('tool_executions', [])
    if recent_tools:
        with st.expander("🔄 Recent Activity", expanded=False):
            # Show last 3 tool executions
            for tool_exec in recent_tools[-3:]:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"🔧 **{tool_exec['tool_name']}**")
                with col2:
                    st.caption(tool_exec.get('timestamp', 'Unknown'))


def render_chat_history():
    """Render the enhanced chat history display with better formatting."""
    if not st.session_state.chat_history:
        # Show welcome message when no chat history
        with st.container():
            st.markdown("""
            <div style="text-align: center; padding: 2rem; background-color: #f0f2f6; border-radius: 10px; margin: 1rem 0;">
                <h3>👋 Welcome to your MCP Playground!</h3>
                <p>Start a conversation by typing a message below. I can help you with various tasks using connected tools.</p>
            </div>
            """, unsafe_allow_html=True)
        return
    
    # Display conversation statistics
    with st.expander("📊 Conversation Stats", expanded=False):
        user_msgs = len([m for m in st.session_state.chat_history if m["role"] == "user"])
        assistant_msgs = len([m for m in st.session_state.chat_history if m["role"] == "assistant"])
        tool_executions = len(st.session_state.get('tool_executions', []))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Your Messages", user_msgs)
        with col2:
            st.metric("AI Responses", assistant_msgs)
        with col3:
            st.metric("Tools Used", tool_executions)
    
    st.divider()
    
    # Display chat messages with enhanced formatting
    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.write(message["content"])
                # Show attachments if any
                attachments_meta = message.get("attachments", [])
                if attachments_meta:
                    with st.expander(f"📎 Attachments ({len(attachments_meta)})", expanded=False):
                        for meta in attachments_meta:
                            st.caption(f"{meta.get('name', 'file')} - {meta.get('type', '')}")
                
                # Show timestamp if available
                if "timestamp" in message:
                    st.caption(f"Sent at {message['timestamp']}")
        
        elif message["role"] == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                # Show any tool executions that happened with this response
                if "tool" in message and message["tool"]:
                    with st.status("🔧 Tool executions for this response", expanded=False, state="complete"):
                        st.code(message['tool'], language="text")
                
                # Display thinking content if available
                if "thinking" in message and message["thinking"]:
                    with st.expander("🧠 View Model Reasoning", expanded=False):
                        st.write("**Model's thinking process:**")
                        st.write(message["thinking"])
                
                # Display the main response
                st.write(message["content"])
                
                # Show timestamp, model info, and response time if available
                if "timestamp" in message:
                    caption_parts = [f"🕒 {message['timestamp']}"]
                    
                    # Add model information
                    if "model_provider" in message and "model_name" in message:
                        model_info = f"{message['model_provider']} - {message['model_name']}"
                        caption_parts.append(f"🤖 {model_info}")
                    
                    # Add response time
                    if "response_time" in message:
                        response_time = message['response_time']
                        if response_time < 1:
                            time_str = f"{response_time*1000:.0f}ms"
                        else:
                            time_str = f"{response_time:.1f}s"
                        caption_parts.append(f"⌛ {time_str}")
                    
                    # Add thinking indicator if thinking content exists
                    if "thinking" in message and message["thinking"]:
                        caption_parts.append("🧠 Reasoning available")
                    
                    st.caption(" • ".join(caption_parts))
                
                # Show related tool executions from session state
                message_tools = get_tools_for_message_index(i)
                if message_tools:
                    with st.expander(f"🔍 View tool details ({len(message_tools)} tools used)", expanded=False):
                        for tool_exec in message_tools:
                            with st.container():
                                st.write(f"**🔧 {tool_exec['tool_name']}**")
                                
                                col1, col2 = st.columns([1, 1])
                                with col1:
                                    if tool_exec.get('input'):
                                        st.write("**Input:**")
                                        if isinstance(tool_exec['input'], dict):
                                            st.json(tool_exec['input'], expanded=False)
                                        else:
                                            st.text(str(tool_exec['input']))
                                
                                with col2:
                                    st.write("**Output:**")
                                    output = tool_exec.get('output', '')
                                    if isinstance(output, str) and len(output) > 200:
                                        st.text(output[:200] + "...")
                                        st.caption("(Output truncated - full output available in tool execution details)")
                                    elif isinstance(output, (dict, list)):
                                        st.json(output, expanded=False)
                                    else:
                                        st.text(str(output))
                                
                                st.caption(f"Executed at {tool_exec.get('timestamp', 'Unknown time')}")
                                st.divider()


def get_tools_for_message_index(message_index: int) -> List[Dict]:
    """Get tool executions that are related to a specific message index."""
    # Check if this specific message has associated tool executions
    if message_index < len(st.session_state.chat_history):
        message = st.session_state.chat_history[message_index]
        
        # Return tool executions that are directly associated with this message
        return message.get('tool_executions', [])
    
    return []


def handle_chat_input():
    """Handle chat input and agent processing."""
    submission = st.chat_input(
        "Type your message here...",
        accept_file="multiple",
        file_type=["pdf", "txt", "md", "png", "jpg", "jpeg", "gif", "webp"]
    )
    if submission:
        # Parse submission (string or dict-like with text/files)
        if isinstance(submission, str):
            text = submission
            files = []
        else:
            # Support attribute and key access
            text = getattr(submission, "text", None)
            if text is None:
                try:
                    text = submission.get("text", "")  # type: ignore[attr-defined]
                except Exception:
                    text = ""
            files = getattr(submission, "files", None)
            if files is None:
                try:
                    files = submission.get("files", [])  # type: ignore[attr-defined]
                except Exception:
                    files = []
        # Build attachments from uploaded files
        attachments = []
        for uf in files or []:
            try:
                data = uf.getvalue() if hasattr(uf, "getvalue") else uf.read()
            except Exception:
                data = b""
            name = getattr(uf, "name", "file") or "file"
            lower = name.lower()
            if any(lower.endswith(e) for e in [".png", ".jpg", ".jpeg", ".gif", ".webp"]):
                kind = "image"
            elif lower.endswith(".pdf"):
                kind = "pdf"
            else:
                kind = "text"
            attachments.append({"name": name, "data": data, "type": kind})

        # Build multimodal content
        provider = st.session_state.get('llm_provider', '')
        model = st.session_state.get('selected_model', '')
        message_content = build_multimodal_human_content(provider, model, text or "", attachments)

        # Add user message to chat history with metadata
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        message_count = len(st.session_state.chat_history) + 1
        message_id = f"msg_{message_count:04d}"

        user_message = {
            "role": "user",
            "content": text or "",
            "timestamp": current_time,
            "message_id": message_id,
            "attachments": [{"name": a.get("name"), "type": a.get("type")} for a in attachments] if attachments else []
        }
        st.session_state.chat_history.append(user_message)
        st.chat_message("user").write(text or "")

        # Check if agent is set up
        if st.session_state.agent is None:
            st.error("Please initialize an agent first (either connect to an MCP server or start chat-only mode)")
        else:
            process_user_message(text or "", message_content)


def process_user_message(user_input: str, message_content: Any):
    """Process user message through the agent with streaming support."""
    with st.chat_message("assistant"):
        # Check if streaming is enabled in session state
        streaming_enabled = st.session_state.get('enable_streaming', True)
        
        if streaming_enabled:
            # Use streaming approach
            process_streaming_response(user_input, message_content)
        else:
            # Use original non-streaming approach
            process_non_streaming_response(user_input, message_content)


def process_streaming_response(user_input: str, message_content: Any):
    """Process user message with enhanced streaming using st.status and st.write_stream with reasoning detection."""
    # Track response timing
    start_time = time.time()
    
    # Prepare agent invocation config
    config = prepare_agent_invocation_config(
        memory_enabled=st.session_state.get('memory_enabled', False),
        thread_id=st.session_state.get('thread_id', 'default')
    )
    
    # Initialize tracking variables
    current_response = ""
    tool_executions = []
    thinking_content = ""
    final_response = ""
    
    # Create containers for dynamic updates
    main_status_container = st.empty()
    
    try:
        # Track if response streaming has started
        response_started = False
        
        # Create async generator processing function
        async def process_streaming():
            nonlocal current_response, tool_executions, thinking_content, final_response, response_started
            
            # Placeholders for streaming content
            thinking_stream_placeholder = None
            response_placeholder = None
            
            # Track different phases
            thinking_phase = True
            tool_phase = False
            reasoning_complete = False
            text_buffer = ""
            thinking_round = 1  # Track which round of thinking we're in
            all_thinking_content = []  # Accumulate all thinking rounds for chat history
            
            # Update main status
            with main_status_container.container():
                with st.status("🤖 Processing your request...", expanded=True) as main_status:
                    main_status.update(label="🧠 **Agent initialized** - Analyzing your request...", state="running")
                    
                    async for event in stream_agent_events(st.session_state.agent, message_content, config):
                        event_type = event.get("event")
                        event_name = event.get("name", "")
                        
                        # Handle different event types with enhanced status updates
                        if event_type == "on_chain_start":
                            if "agent" in event_name.lower():
                                main_status.update(label="🔍 **Agent reasoning** - Planning approach...", state="running")
                                thinking_phase = True
                        
                        elif event_type == "on_tool_start":
                            # Tool execution started
                            tool_name = event.get("name", "Unknown")
                            tool_input = event.get("data", {}).get("input", {})
                            
                            if thinking_phase:
                                main_status.update(label="✅ **Planning complete** - Ready to execute tools", state="complete")
                                thinking_phase = False
                                tool_phase = True
                            
                            # Track tool execution without nested status
                            main_status.update(label=f"🔧 **Starting tool:** {tool_name}", state="running")
                            if tool_input:
                                st.write(f"🔧 Using tool: {tool_name}")
                        
                        elif event_type == "on_tool_end":
                            # Tool execution completed
                            tool_name = event.get("name", "Unknown")
                            tool_output = event.get("data", {}).get("output", "")
                            tool_input = event.get("data", {}).get("input", {})
                            
                            main_status.update(label=f"✅ **Completed tool:** {tool_name}", state="complete")
                            st.write(f"🔧 Finished using tool: {tool_name}")
                            
                            # Store tool execution data
                            tool_executions.append({
                                "tool_name": tool_name,
                                "input": tool_input,
                                "output": tool_output,
                                "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            })
                            
                            # Reset reasoning detection for potential post-tool thinking
                            reasoning_complete = False
                            thinking_stream_placeholder = None
                            thinking_round += 1
                            # Clear current thinking content for new round (but keep accumulated)
                            thinking_content = ""
                        
                        elif event_type == "on_chat_model_start":
                            if tool_phase:
                                main_status.update(label="🎯 **Tool execution complete** - Generating response...", state="running")
                                tool_phase = False
                        
                        elif event_type == "on_chat_model_stream":
                            # Handle streaming tokens from the chat model
                            chunk = event.get("data", {}).get("chunk", {})
                            if hasattr(chunk, 'content') and chunk.content:
                                chunk_text = coerce_content_to_text(chunk.content)
                                if not chunk_text:
                                    continue
                                # Add to text buffer for reasoning detection
                                text_buffer += chunk_text
                                
                                # Check for reasoning content
                                reasoning_info = detect_reasoning_in_stream(text_buffer, thinking_round)
                                
                                if reasoning_info['has_thinking'] and not reasoning_complete:
                                    # We're in thinking mode - show thinking content
                                    current_thinking = reasoning_info['thinking_content']
                                    
                                    # Display thinking in status container
                                    if thinking_stream_placeholder is None:
                                        if thinking_round == 1:
                                            main_status.update(label="🧠 **AI is thinking...**", state="running")
                                        else:
                                            main_status.update(label=f"🧠 **AI is thinking** (round {thinking_round})...", state="running")
                                        thinking_stream_placeholder = st.empty()
                                    
                                    with thinking_stream_placeholder:
                                        st.text(f"🧠 {current_thinking}")
                                    
                                    # Check if thinking is complete
                                    if reasoning_info['thinking_complete']:
                                        thinking_content = reasoning_info['thinking_content']
                                        current_response = reasoning_info['response_content']
                                        reasoning_complete = True
                                        
                                        # Add this thinking round to accumulated content
                                        if thinking_round > 1:
                                            all_thinking_content.append(f"**Round {thinking_round}:**\n{thinking_content}")
                                        else:
                                            all_thinking_content.append(thinking_content)
                                        
                                        # Show completion of this thinking round
                                        if thinking_round > 1:
                                            st.badge(f"**Thinking round {thinking_round} complete!**", icon="✅")
                                        else:
                                            st.badge("**Thinking complete!**", icon="✅")
                                
                                elif reasoning_complete:
                                    # Thinking is complete, accumulate and stream response content
                                    if reasoning_info['response_content'] != current_response:
                                        current_response = reasoning_info['response_content']
                                        
                                        # Start streaming response inside status container
                                        if not response_started:
                                            main_status.update(label="💬 **Streaming response...**", state="running")
                                            response_started = True
                                            st.divider()  # Add separator after thinking
                                            st.write("💬 **Final Response:**")
                                            response_placeholder = st.empty()
                                        
                                        # Update response in real-time inside status container
                                        if response_started:
                                            response_placeholder.write(current_response)
                                else:
                                    # No reasoning detected, accumulate and stream normal response
                                    current_response += chunk_text
                                    
                                    # Start streaming response inside status container
                                    if not response_started:
                                        main_status.update(label="💬 **Streaming response...**", state="running")
                                        response_started = True
                                        st.write("💬 **Final Response:**")
                                        response_placeholder = st.empty()
                                    
                                    # Update response in real-time inside status container
                                    if response_started:
                                        response_placeholder.write(current_response)
                        
                        elif event_type == "on_chat_model_end":
                            # Model finished generating
                            final_response_event = event.get("data", {}).get("output", {})
                            
                            # If we didn't get content through streaming, use the final response
                            if not current_response and final_response_event:
                                if hasattr(final_response_event, 'content'):
                                    full_content = final_response_event.content
                                elif isinstance(final_response_event, str):
                                    full_content = final_response_event
                                else:
                                    full_content = str(final_response_event)
                                
                                # Parse the full content for reasoning
                                full_text = coerce_content_to_text(full_content)
                                parsed_content = parse_reasoning_content(full_text)
                                
                                if parsed_content['thinking']:
                                    thinking_content = parsed_content['thinking']
                                    current_response = parsed_content['response']
                                    
                                    # Show thinking if we haven't already
                                    if not reasoning_complete:
                                        st.write("🧠 **Model reasoning detected**")
                                        
                                        # Display thinking content inside the current status container
                                        st.text(f"🧠 {thinking_content}")
                                        
                                        # Add to accumulated thinking content
                                        all_thinking_content.append(thinking_content)
                                        
                                        reasoning_complete = True
                                else:
                                    current_response = full_text
                    
                    # Update main status when processing is complete
                    if current_response:
                        main_status.update(
                            label="✅ Response complete",
                            state="complete"
                        )
                        main_status.update(label="📝 **Processing complete** - Streaming response...", state="complete")
            
            return current_response, tool_executions, all_thinking_content
        
        # Process the streaming using run_async
        result = run_async(process_streaming())
        if result:
            final_response, final_tool_executions, final_thinking_list = result
            current_response = final_response or current_response
            tool_executions = final_tool_executions or tool_executions
            # Combine all thinking rounds into a single string for chat history
            thinking_content = "\n\n".join(final_thinking_list) if final_thinking_list else ""
            
    except Exception as e:
        # Update main status to show error
        with main_status_container.container():
            with st.status("❌ Processing failed", expanded=True, state="error"):
                st.error(f"Streaming failed: {str(e)}")
                st.info("🔄 Falling back to non-streaming mode...")
        
        process_non_streaming_response(user_input, message_content)
        return
    
    # Store the response in chat history regardless of streaming status
    if current_response:
        # Response is already displayed inside the status container
        # No need for external display
        
        # Always store the response in chat history with timing and model info
        end_time = time.time()
        response_time = end_time - start_time
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        message_count = len(st.session_state.chat_history) + 1
        message_id = f"msg_{message_count:04d}"
        
        # Get model information
        model_provider = st.session_state.get('llm_provider', 'Unknown')
        model_name = st.session_state.get('selected_model', 'Unknown')
        
        assistant_message = {
            "role": "assistant", 
            "content": current_response,
            "timestamp": current_time,
            "message_id": message_id,
            "model_provider": model_provider,
            "model_name": model_name,
            "response_time": response_time
        }
        
        # Add thinking content if available
        if thinking_content:
            assistant_message["thinking"] = thinking_content
        
        st.session_state.chat_history.append(assistant_message)
        
        # Auto-save conversation if persistent storage is enabled
        handle_auto_save(current_response)
    
    # Handle tool executions summary
    if tool_executions:
        # Associate tool executions with the last assistant message
        if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "assistant":
            st.session_state.chat_history[-1]["tool_executions"] = tool_executions
        
        # Also add to global tool executions for the test tools tab
        existing_executions = st.session_state.get('tool_executions', [])
        new_executions = []
        
        for execution in tool_executions:
            # Check if this execution already exists
            if not any(
                ex['timestamp'] == execution['timestamp'] and 
                ex['tool_name'] == execution['tool_name']
                for ex in existing_executions
            ):
                new_executions.append(execution)
        
        if new_executions:
            if 'tool_executions' not in st.session_state:
                st.session_state.tool_executions = []
            st.session_state.tool_executions.extend(new_executions)
    
    st.rerun()


def process_non_streaming_response(user_input: str, message_content: Any):
    """Process user message with non-streaming response (original implementation) with reasoning detection."""
    # Track response timing
    start_time = time.time()
    
    # Initialize tracking variables
    current_response = ""
    tool_executions = []
    thinking_content = ""
    
    # Create containers for dynamic updates
    main_status_container = st.empty()
    
    try:
        # Prepare agent invocation config
        config = prepare_agent_invocation_config(
            memory_enabled=st.session_state.get('memory_enabled', False),
            thread_id=st.session_state.get('thread_id', 'default')
        )
        
        # Update main status
        with main_status_container.container():
            with st.status("🤖 Processing your request...", expanded=True) as main_status:
                st.write("🧠 **Agent initialized** - Analyzing your request...")
                
                # Run the agent with context isolation and shorter timeout
                if config:
                    # For memory-enabled agents, use safe async call with reasonable timeout
                    response = safe_async_call(
                        st.session_state.agent.ainvoke({"messages": [HumanMessage(content=message_content)]}, config),
                        "Failed to process message with memory",
                        timeout=600.0  # 10 minutes timeout for chat responses
                    )
                else:
                    # For agents without memory, use safe async call with shorter timeout
                    response = safe_async_call(
                        run_agent(st.session_state.agent, message_content),
                        "Failed to process message",
                        timeout=600.0  # 10 minutes timeout for simple chat
                    )
                
                if response is None:
                    main_status.update(label="❌ Processing failed", state="error")
                    st.error("Failed to get response from agent. Please try again.")
                    return
                
                st.write("🔍 **Agent reasoning** - Processing your request...")
                
                # Process response
                tool_executions = extract_tool_executions_from_response(response)
                assistant_response = extract_assistant_response(response)
                
                # Handle tool executions if any
                if tool_executions:
                    st.write("🔧 **Tool executions detected**")
                    for execution in tool_executions:
                        st.write(f"✅ **Completed tool:** {execution['tool_name']}")
                
                if assistant_response:
                    # Parse the response for reasoning content
                    parsed_content = parse_reasoning_content(assistant_response)
                    
                    # Display thinking if present inside status container
                    if parsed_content['thinking']:
                        st.write("🧠 **AI is thinking...**")
                        st.text(f"💭 {parsed_content['thinking']}")
                        thinking_content = parsed_content['thinking']
                    
                    # Set the final response
                    current_response = parsed_content['response'] if parsed_content['response'] else assistant_response
                    
                    # Display response inside status container
                    if thinking_content:
                        st.divider()  # Add separator after thinking
                    st.write("💬 **Final Response:**")
                    st.write(current_response)
                    
                    # Update main status when processing is complete
                    main_status.update(
                        label="✅ Response complete",
                        state="complete"
                    )
                else:
                    main_status.update(label="⚠️ No response content", state="error")
                    st.warning("No response content found.")
                    return
            
            # Add to chat history with metadata, timing, and model info
            end_time = time.time()
            response_time = end_time - start_time
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            message_count = len(st.session_state.chat_history) + 1
            message_id = f"msg_{message_count:04d}"
            
            # Get model information
            model_provider = st.session_state.get('llm_provider', 'Unknown')
            model_name = st.session_state.get('selected_model', 'Unknown')
            
            assistant_message = {
                "role": "assistant", 
                "content": current_response,
                "timestamp": current_time,
                "message_id": message_id,
                "model_provider": model_provider,
                "model_name": model_name,
                "response_time": response_time
            }
            
            # Add thinking content if available
            if thinking_content:
                assistant_message["thinking"] = thinking_content
            
            st.session_state.chat_history.append(assistant_message)
            
            # Auto-save conversation if persistent storage is enabled
            handle_auto_save(current_response)
        
        # Handle tool executions summary (consistent with streaming)
        if tool_executions:
            # Associate tool executions with the last assistant message
            if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "assistant":
                st.session_state.chat_history[-1]["tool_executions"] = tool_executions
            
            # Also add to global tool executions for the test tools tab
            existing_executions = st.session_state.get('tool_executions', [])
            new_executions = []
            
            for execution in tool_executions:
                # Check if this execution already exists (by timestamp and tool name)
                if not any(
                    ex['timestamp'] == execution['timestamp'] and 
                    ex['tool_name'] == execution['tool_name']
                    for ex in existing_executions
                ):
                    new_executions.append(execution)
            
            if new_executions:
                if 'tool_executions' not in st.session_state:
                    st.session_state.tool_executions = []
                st.session_state.tool_executions.extend(new_executions)
                
                # Show execution summary with enhanced status (consistent with streaming)
                with st.status(f"📊 Tool Execution Summary", expanded=False, state="complete"):
                    st.write(f"**Executed {len(new_executions)} tool(s) successfully:**")
                    for execution in new_executions:
                        st.write(f"• **{execution['tool_name']}** at {execution['timestamp']}")
                
    except Exception as e:
        # Update main status to show error
        with main_status_container.container():
            with st.status("❌ Processing failed", expanded=True, state="error"):
                st.error(f"Non-streaming processing failed: {str(e)}")
                st.code(traceback.format_exc(), language="python")
    
    st.rerun()


def render_attachment_uploader():
    """Render file uploader and preview; store attachments in session state."""
    with st.expander("📎 Attach files (PDF, TXT, images)", expanded=False):
        uploaded = st.file_uploader(
            "Choose files",
            type=["pdf", "txt", "md", "png", "jpg", "jpeg", "gif", "webp"],
            accept_multiple_files=True,
            key="chat_file_uploader"
        )
        attachments = []
        if uploaded:
            for uf in uploaded:
                try:
                    data = uf.read()
                except Exception:
                    data = b""
                name = getattr(uf, "name", "file")
                ext = (name or "").lower()
                if any(ext.endswith(e) for e in [".png", ".jpg", ".jpeg", ".gif", ".webp"]):
                    kind = "image"
                elif ext.endswith(".pdf"):
                    kind = "pdf"
                else:
                    kind = "text"
                attachments.append({"name": name, "data": data, "type": kind})
        # Save to session state (overwrite to reflect current selection)
        st.session_state.chat_attachments = attachments

        # Preview summary
        if attachments:
            cols = st.columns([3, 1])
            with cols[0]:
                st.write("Selected:")
                for a in attachments[:10]:
                    st.caption(f"{a['name']} ({a['type']})")
                if len(attachments) > 10:
                    st.caption(f"...and {len(attachments) - 10} more")
            with cols[1]:
                if st.button("Clear", key="clear_attachments"):
                    st.session_state.chat_attachments = []
                    st.rerun()


def handle_auto_save(assistant_response: str):
    """Handle automatic saving to persistent storage."""
    if (st.session_state.get('memory_enabled', False) and 
        st.session_state.get('memory_type') == "Persistent (Cross-session)" and
        hasattr(st.session_state, 'persistent_storage')):
        
        try:
            thread_id = st.session_state.get('thread_id', 'default')
            # Use the new synchronous save method for better performance
            st.session_state.persistent_storage.save_conversation_sync(
                thread_id=thread_id,
                chat_history=st.session_state.chat_history
            )
        except Exception as e:
            # Don't show error to user for auto-save failures, but log it
            pass


def apply_message_trimming():
    """Apply message trimming if memory is enabled and we have too many messages."""
    if st.session_state.get('memory_enabled', False):
        max_messages = st.session_state.get('max_messages', 100)
        if len(st.session_state.chat_history) > max_messages:
            # Trim older messages but keep some context
            keep_messages = max_messages // 2
            st.session_state.chat_history = st.session_state.chat_history[-keep_messages:]


def handle_chat_error(error: Exception):
    """Handle errors during chat processing with improved messaging."""
    formatted_error = format_error_message(error)
    
    # Check for specific error types for better user guidance
    error_msg = str(error)
    if "All connection attempts failed" in error_msg:
        st.error("⚠️ Could not connect to Ollama. Please make sure Ollama is running by executing 'ollama serve' in a terminal.")
        st.info("To start Ollama, open a terminal/command prompt and run: `ollama serve`")
    elif "cannot enter context" in error_msg or "already entered" in error_msg:
        st.error("🔄 Context conflict detected. Retrying with isolated context...")
        st.info("💡 This can happen with external MCP servers. The system will handle this automatically.")
    elif "timeout" in error_msg.lower():
        st.error("⏱️ Request timed out. The server may be overloaded or unreachable.")
        st.info("💡 Try again in a moment, or check your MCP server connection.")
    else:
        st.error(f"❌ Error processing your request: {formatted_error}")
    
    # Show technical details in expandable section
    with st.expander("🔧 Technical Details"):
        st.code(traceback.format_exc(), language="python")


def render_test_tools_tab():
    """Render the tool testing interface tab."""
    st.header("🔧 Test Tools Individually")
    
    if not st.session_state.tools:
        st.warning("⚠️ No tools available. Please connect to an MCP server first.")
        st.info("Go to the sidebar to connect to an MCP server, then return to this tab to test tools.")
        return
    
    # Tool selection and information
    selected_tool = render_tool_selection()
    if not selected_tool:
        return
    
    # Tool parameters and execution
    tool_params, missing_required = render_tool_parameters_form(selected_tool)
    
    # Tool execution controls
    render_tool_execution_controls(selected_tool, tool_params, missing_required)
    
    # Test results display
    render_test_results(selected_tool.name)
    
    # Testing summary
    render_testing_summary()


def render_tool_selection():
    """Render tool selection interface."""
    st.subheader("Select Tool to Test")
    
    tool_names = [tool.name for tool in st.session_state.tools]
    selected_tool_name = st.selectbox(
        "Choose a tool:",
        options=tool_names,
        key="test_tool_selector"
    )
    
    if not selected_tool_name:
        return None
    
    selected_tool = next((tool for tool in st.session_state.tools if tool.name == selected_tool_name), None)
    
    if selected_tool:
        render_selected_tool_info(selected_tool, selected_tool_name)
    
    return selected_tool


def render_selected_tool_info(selected_tool, selected_tool_name):
    """Render information about the selected tool."""
    st.subheader("Tool Information")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write(f"**Name:** {selected_tool.name}")
        st.write(f"**Description:** {selected_tool.description}")
    
    with col2:
        # Tool execution statistics
        tool_stats = st.session_state.get('tool_test_stats', {})
        if selected_tool_name in tool_stats:
            stats = tool_stats[selected_tool_name]
            st.metric("Tests Run", stats.get('count', 0))
            st.metric("Success Rate", f"{stats.get('success_rate', 0):.1f}%")
            st.metric("Avg Time", f"{stats.get('avg_time', 0):.2f}s")


def render_tool_parameters_form(tool):
    """Render dynamic form for tool parameters."""
    st.subheader("Tool Parameters")
    
    tool_params = {}
    required_params = []
    
    if hasattr(tool, 'args_schema') and tool.args_schema:
        schema = tool.args_schema
        if hasattr(schema, 'schema'):
            schema_dict = schema.schema()
            properties = schema_dict.get('properties', {})
            required_params = schema_dict.get('required', [])
        else:
            properties = schema.get('properties', {})
            required_params = schema.get('required', [])
        
        if properties:
            st.write("Fill in the parameters below:")
            tool_params = render_parameter_inputs(properties, required_params)
        else:
            st.info("This tool doesn't require any parameters.")
    else:
        st.info("This tool doesn't require any parameters.")
    
    # Validate required parameters
    missing_required = []
    for req_param in required_params:
        if req_param in tool_params:
            value = tool_params[req_param]
            if value is None or value == "" or (isinstance(value, list) and len(value) == 0):
                missing_required.append(req_param)
    
    if missing_required:
        st.warning(f"⚠️ Required parameters missing: {', '.join(missing_required)}")
    
    return tool_params, missing_required


def render_parameter_inputs(properties, required_params):
    """Render input widgets for tool parameters."""
    tool_params = {}
    
    for param_name, param_info in properties.items():
        param_type = param_info.get('type', 'string')
        param_title = param_info.get('title', param_name)
        param_description = param_info.get('description', '')
        param_default = param_info.get('default', None)
        is_required = param_name in required_params
        
        label = f"{param_title}"
        if is_required:
            label += " *"
        
        # Create appropriate input widget based on type
        if param_type == 'integer':
            value = st.number_input(
                label,
                value=param_default if param_default is not None else 0,
                step=1,
                help=param_description,
                key=f"test_param_{param_name}"
            )
        elif param_type == 'number':
            value = st.number_input(
                label,
                value=float(param_default) if param_default is not None else 0.0,
                step=0.1,
                help=param_description,
                key=f"test_param_{param_name}"
            )
        elif param_type == 'boolean':
            value = st.checkbox(
                label,
                value=param_default if param_default is not None else False,
                help=param_description,
                key=f"test_param_{param_name}"
            )
        elif param_type == 'array':
            text_value = st.text_area(
                label,
                value="",
                help=f"{param_description}\n(Enter items separated by new lines)",
                key=f"test_param_{param_name}"
            )
            value = [item.strip() for item in text_value.split('\n') if item.strip()] if text_value else []
        else:
            value = st.text_input(
                label,
                value=param_default if param_default is not None else "",
                help=param_description,
                key=f"test_param_{param_name}"
            )
        
        tool_params[param_name] = value
    
    return tool_params


def render_tool_execution_controls(tool, tool_params, missing_required):
    """Render tool execution controls."""
    st.subheader("Execute Tool")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🚀 Run Tool", type="primary", disabled=len(missing_required) > 0):
            execute_tool_test(tool, tool_params)
    
    with col2:
        if st.button("🗑️ Clear Results"):
            clear_test_results()
    
    with col3:
        if st.button("📋 Copy Parameters as JSON"):
            params_json = json.dumps(tool_params, indent=2)
            st.code(params_json, language="json")


def execute_tool_test(tool, tool_params):
    """Execute a tool test and record results."""
    with st.spinner("Executing tool..."):
        start_time = datetime.datetime.now()
        
        try:
            result = run_async(run_tool(tool, **tool_params))
            end_time = datetime.datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Store successful result
            store_test_result(tool.name, tool_params, result, True, execution_time, start_time)
            update_test_statistics(tool.name, True, execution_time)
            
            st.success(f"✅ Tool executed successfully in {execution_time:.2f} seconds!")
            st.rerun()
            
        except Exception as e:
            end_time = datetime.datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Store error result
            store_test_result(tool.name, tool_params, None, False, execution_time, start_time, str(e))
            update_test_statistics(tool.name, False, execution_time)
            
            st.error(f"❌ Tool execution failed: {str(e)}")
            st.code(traceback.format_exc(), language="python")


def store_test_result(tool_name, parameters, result, success, execution_time, timestamp, error=None):
    """Store test result in session state."""
    if 'tool_test_results' not in st.session_state:
        st.session_state.tool_test_results = []
    
    test_result = {
        'tool_name': tool_name,
        'parameters': parameters.copy(),
        'result': result,
        'success': success,
        'execution_time': execution_time,
        'timestamp': timestamp.isoformat()
    }
    
    if error:
        test_result['error'] = error
    
    st.session_state.tool_test_results.insert(0, test_result)


def update_test_statistics(tool_name, success, execution_time):
    """Update test statistics for a tool."""
    if 'tool_test_stats' not in st.session_state:
        st.session_state.tool_test_stats = {}
    
    if tool_name not in st.session_state.tool_test_stats:
        st.session_state.tool_test_stats[tool_name] = {
            'count': 0,
            'successes': 0,
            'total_time': 0.0
        }
    
    stats = st.session_state.tool_test_stats[tool_name]
    stats['count'] += 1
    if success:
        stats['successes'] += 1
    stats['total_time'] += execution_time
    stats['success_rate'] = (stats['successes'] / stats['count']) * 100
    stats['avg_time'] = stats['total_time'] / stats['count']


def clear_test_results():
    """Clear all test results."""
    if 'tool_test_results' in st.session_state:
        st.session_state.tool_test_results = []
    if 'tool_test_stats' in st.session_state:
        st.session_state.tool_test_stats = {}
    st.success("Results cleared!")
    st.rerun()


def render_test_results(tool_name):
    """Render test results for the current tool."""
    if 'tool_test_results' not in st.session_state or not st.session_state.tool_test_results:
        return
    
    st.subheader("Test Results")
    
    # Filter results for current tool
    current_tool_results = [r for r in st.session_state.tool_test_results if r['tool_name'] == tool_name]
    
    if current_tool_results:
        render_latest_result(current_tool_results[0])
        render_result_history(current_tool_results[1:] if len(current_tool_results) > 1 else [])
    else:
        st.info("No test results for this tool yet.")


def render_latest_result(result):
    """Render the latest test result."""
    if result['success']:
        st.success("✅ Latest Result")
        with st.expander("View Result Details", expanded=True):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write("**Output:**")
                if isinstance(result['result'], str):
                    st.text(result['result'])
                else:
                    st.json(result['result'])
            with col2:
                st.write("**Execution Info:**")
                st.write(f"Time: {result['execution_time']:.2f}s")
                st.write(f"Timestamp: {result['timestamp']}")
                
                if result['parameters']:
                    st.write("**Parameters:**")
                    st.json(result['parameters'])
    else:
        st.error("❌ Latest Result")
        with st.expander("View Error Details", expanded=True):
            st.write("**Error:**")
            st.code(result['error'])
            st.write(f"**Execution Time:** {result['execution_time']:.2f}s")
            if result['parameters']:
                st.write("**Parameters:**")
                st.json(result['parameters'])


def render_result_history(history_results):
    """Render historical test results."""
    if not history_results:
        return
    
    with st.expander(f"Previous Results ({len(history_results)})"):
        for i, result in enumerate(history_results, 1):
            status = "✅" if result['success'] else "❌"
            time_str = datetime.datetime.fromisoformat(result['timestamp']).strftime("%H:%M:%S")
            
            st.write(f"**{status} Test #{i + 1}** - {time_str} ({result['execution_time']:.2f}s)")
            
            if result['success']:
                if isinstance(result['result'], str):
                    display_text = result['result'][:200] + "..." if len(str(result['result'])) > 200 else result['result']
                    st.text(display_text)
                else:
                    st.json(result['result'])
            else:
                st.error(f"Error: {result['error']}")
            
            st.divider()


def render_testing_summary():
    """Render overall testing summary."""
    if 'tool_test_results' not in st.session_state or not st.session_state.tool_test_results:
        return
    
    st.subheader("Testing Summary")
    
    col1, col2, col3 = st.columns(3)
    
    total_tests = len(st.session_state.tool_test_results)
    successful_tests = len([r for r in st.session_state.tool_test_results if r['success']])
    overall_success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
    
    with col1:
        st.metric("Total Tests", total_tests)
        st.metric("Success Rate", f"{overall_success_rate:.1f}%")
    
    with col2:
        if st.session_state.tool_test_results:
            avg_execution_time = sum(r['execution_time'] for r in st.session_state.tool_test_results) / len(st.session_state.tool_test_results)
            tools_tested = len(set(r['tool_name'] for r in st.session_state.tool_test_results))
            
            st.metric("Avg Execution Time", f"{avg_execution_time:.2f}s")
            st.metric("Tools Tested", tools_tested)
    
    with col3:
        if st.button("📁 Export Test Results"):
            export_test_results(total_tests, successful_tests, overall_success_rate)


def export_test_results(total_tests, successful_tests, overall_success_rate):
    """Export test results to JSON."""
    if st.session_state.tool_test_results:
        avg_execution_time = sum(r['execution_time'] for r in st.session_state.tool_test_results) / len(st.session_state.tool_test_results)
        tools_tested = len(set(r['tool_name'] for r in st.session_state.tool_test_results))
        
        export_data = {
            'export_timestamp': datetime.datetime.now().isoformat(),
            'summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'success_rate': overall_success_rate,
                'avg_execution_time': avg_execution_time,
                'tools_tested': tools_tested
            },
            'detailed_results': st.session_state.tool_test_results,
            'statistics': st.session_state.tool_test_stats
        }
        
        json_str, filename = create_download_data(export_data, "tool_test_results")
        st.download_button(
            label="Download JSON Report",
            data=json_str,
            file_name=filename,
            mime="application/json"
        )


def render_memory_tab():
    """Render the memory management tab."""
    st.header("🧠 Memory Management")
    
    # Memory status overview
    render_memory_status_overview()
    st.divider()
    
    # Enhanced memory configuration
    render_memory_configuration_section()
    
    # Database management for persistent storage
    render_database_management_section()
    
    # Memory actions
    render_memory_actions_section()
    
    # Conversation history viewer
    render_conversation_history_section()
    
    # Memory tips
    render_memory_tips()


def render_memory_status_overview():
    """Render memory status overview."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        memory_enabled = st.session_state.get('memory_enabled', False)
        if memory_enabled:
            st.success("Memory: Enabled")
        else:
            st.info("Memory: Disabled")
    
    with col2:
        if memory_enabled:
            memory_type = st.session_state.get('memory_type', 'Short-term (Session)')
            thread_id = st.session_state.get('thread_id', 'default')
            st.info(f"Type: {memory_type.split()[0]}")
            st.info(f"Thread: {thread_id}")
        else:
            st.info("Type: N/A")
            st.info("Thread: N/A")
    
    with col3:
        chat_length = len(st.session_state.get('chat_history', []))
        st.metric("Messages", chat_length)


def render_memory_configuration_section():
    """Render enhanced memory configuration section."""
    st.subheader("Memory Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        render_memory_toggle_and_type()
    
    with col2:
        render_memory_limits_and_storage()


def render_memory_toggle_and_type():
    """Render memory toggle and type selection."""
    new_memory_enabled = st.toggle(
        "Enable Memory",
        value=st.session_state.get('memory_enabled', False),
        help="Enable or disable conversation memory"
    )
    
    if new_memory_enabled != st.session_state.get('memory_enabled', False):
        st.session_state.memory_enabled = new_memory_enabled
        if 'agent' in st.session_state:
            st.session_state.agent = None
        st.success(f"Memory {'enabled' if new_memory_enabled else 'disabled'}. Please reconnect to MCP server.")
        st.rerun()
    
    if new_memory_enabled:
        new_memory_type = st.selectbox(
            "Storage Type",
            options=["Short-term (Session)", "Persistent (Cross-session)"],
            index=0 if st.session_state.get('memory_type', 'Short-term (Session)') == 'Short-term (Session)' else 1,
            help="Choose memory persistence level"
        )
        
        if new_memory_type != st.session_state.get('memory_type', 'Short-term (Session)'):
            st.session_state.memory_type = new_memory_type
            if 'agent' in st.session_state:
                st.session_state.agent = None
            st.info(f"Memory type changed to: {new_memory_type}. Please reconnect to MCP server.")
            st.rerun()
        
        new_thread_id = st.text_input(
            "Thread ID",
            value=st.session_state.get('thread_id', 'default'),
            help="Unique identifier for conversation thread"
        )
        
        if new_thread_id != st.session_state.get('thread_id', 'default'):
            st.session_state.thread_id = new_thread_id
            st.session_state.chat_history = []
            # Load conversation messages if persistent storage is enabled
            if (st.session_state.get('memory_type') == "Persistent (Cross-session)" and
                hasattr(st.session_state, 'persistent_storage')):
                try:
                    loaded_messages = st.session_state.persistent_storage.load_conversation_messages(new_thread_id)
                    if loaded_messages:
                        st.session_state.chat_history = loaded_messages
                        st.info(f"Loaded {len(loaded_messages)} messages for thread: {new_thread_id}")
                    else:
                        st.info(f"Started new thread: {new_thread_id}")
                except Exception as e:
                    st.warning(f"Could not load conversation history: {str(e)}")
            else:
                st.info(f"Switched to thread: {new_thread_id}")
            st.rerun()


def render_memory_limits_and_storage():
    """Render memory limits and storage information."""
    if st.session_state.get('memory_enabled', False):
        max_messages = st.number_input(
            "Maximum Messages",
            min_value=10,
            max_value=1000,
            value=st.session_state.get('max_messages', 100),
            help="Maximum number of messages to keep in memory"
        )
        st.session_state.max_messages = max_messages
        
        memory_type = st.session_state.get('memory_type', 'Short-term (Session)')
        if memory_type == "Persistent (Cross-session)":
            if 'persistent_storage' not in st.session_state:
                st.session_state.persistent_storage = PersistentStorageManager()
            
            db_stats = st.session_state.persistent_storage.get_database_stats()
            st.info("📊 Database Statistics")
            st.text(f"Conversations: {db_stats.get('conversation_count', 0)}")
            st.text(f"Total Messages: {db_stats.get('total_messages', 0)}")
            st.text(f"Size: {db_stats.get('database_size_mb', 0)} MB")


def render_database_management_section():
    """Render database management section for persistent storage."""
    if (st.session_state.get('memory_enabled', False) and 
        st.session_state.get('memory_type') == "Persistent (Cross-session)" and 
        hasattr(st.session_state, 'persistent_storage')):
        
        st.subheader("📚 Conversation Database")
        render_database_actions()
        render_stored_conversations()


def render_database_actions():
    """Render database action buttons."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("💾 Save Current", type="primary"):
            save_current_conversation()
    
    with col2:
        if st.button("🔄 Refresh List"):
            st.rerun()
    
    with col3:
        if st.button("📊 Database Stats"):
            stats = st.session_state.persistent_storage.get_database_stats()
            st.json(stats)
    
    with col4:
        if st.button("🗑️ Clear All"):
            if st.checkbox("Confirm deletion", key="confirm_clear_all"):
                st.warning("Clear all functionality needs to be implemented")


def save_current_conversation():
    """Save current conversation to database."""
    if st.session_state.chat_history:
        title = None
        for msg in st.session_state.chat_history[:3]:
            if msg.get('role') == 'user':
                content = msg.get('content', '')
                title = content[:50] + "..." if len(content) > 50 else content
                break
        
        thread_id = st.session_state.get('thread_id', 'default')
        st.session_state.persistent_storage.update_conversation_metadata(
            thread_id=thread_id,
            title=title,
            message_count=len(st.session_state.chat_history),
            last_message=st.session_state.chat_history[-1].get('content', '') if st.session_state.chat_history else ''
        )
        st.success("Conversation saved to database!")
        st.rerun()
    else:
        st.warning("No conversation to save")


def render_stored_conversations():
    """Render list of stored conversations."""
    conversations = st.session_state.persistent_storage.list_conversations()
    if conversations:
        st.write(f"**Stored Conversations ({len(conversations)})**")
        
        for i, conv in enumerate(conversations):
            with st.expander(f"📄 {conv.get('title', conv['thread_id'])} ({conv.get('message_count', 0)} messages)"):
                render_conversation_details(conv, i)
    else:
        st.info("No conversations stored yet. Start chatting and save your conversations!")


def render_conversation_details(conv, index):
    """Render details for a single stored conversation."""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write(f"**Thread ID:** {conv['thread_id']}")
        st.write(f"**Created:** {conv.get('created_at', 'Unknown')}")
        st.write(f"**Updated:** {conv.get('updated_at', 'Unknown')}")
        st.write(f"**Messages:** {conv.get('message_count', 0)}")
        if conv.get('last_message'):
            last_msg = conv['last_message']
            if len(last_msg) > 100:
                last_msg = last_msg[:100] + "..."
            st.write(f"**Last Message:** {last_msg}")
    
    with col2:
        if st.button("📂 Load", key=f"load_detailed_{index}"):
            st.session_state.thread_id = conv['thread_id']
            st.session_state.chat_history = []
            # Load conversation messages from database
            if hasattr(st.session_state, 'persistent_storage'):
                try:
                    loaded_messages = st.session_state.persistent_storage.load_conversation_messages(conv['thread_id'])
                    if loaded_messages:
                        st.session_state.chat_history = loaded_messages
                except Exception as e:
                    st.warning(f"Could not load conversation history: {str(e)}")
            st.success(f"Loaded: {conv['thread_id']}")
            st.rerun()
        
        if st.button("📤 Export", key=f"export_detailed_{index}"):
            export_conversation(conv['thread_id'], index)
        
        if st.button("🗑️ Delete", key=f"delete_detailed_{index}"):
            delete_conversation(conv['thread_id'])


def export_conversation(thread_id, index):
    """Export a specific conversation."""
    export_data = st.session_state.persistent_storage.export_conversation(thread_id)
    if export_data:
        json_str, filename = create_download_data(export_data, f"conversation_{thread_id}")
        st.download_button(
            label="📁 Download",
            data=json_str,
            file_name=filename,
            mime="application/json",
            key=f"download_detailed_{index}"
        )


def delete_conversation(thread_id):
    """Delete a specific conversation."""
    if st.session_state.persistent_storage.delete_conversation(thread_id):
        st.success("Conversation deleted")
        st.rerun()
    else:
        st.error("Failed to delete conversation")


def render_memory_actions_section():
    """Render memory actions section."""
    if st.session_state.get('memory_enabled', False):
        st.subheader("Memory Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            render_clear_thread_action()
        
        with col2:
            render_reset_memory_action()
        
        with col3:
            render_export_current_action()
        
        with col4:
            render_import_memory_action()


def render_clear_thread_action():
    """Render clear current thread action."""
    if st.button("🗑️ Clear Current Thread", type="primary"):
        st.session_state.chat_history = []
        if hasattr(st.session_state, 'checkpointer') and st.session_state.checkpointer:
            try:
                thread_id = st.session_state.get('thread_id', 'default')
                st.success(f"Cleared memory for thread: {thread_id}")
            except Exception as e:
                st.error(f"Error clearing memory: {str(e)}")
        else:
            st.success("Chat history cleared!")
        st.rerun()


def render_reset_memory_action():
    """Render reset all memory action."""
    if st.button("🔄 Reset All Memory"):
        st.session_state.chat_history = []
        st.session_state.checkpointer = None
        st.session_state.agent = None
        st.success("All memory reset! Please reconnect to MCP server.")
        st.rerun()


def render_export_current_action():
    """Render export current conversation action."""
    if st.button("💾 Export Current"):
        if st.session_state.chat_history:
            memory_export = format_chat_history_for_export(st.session_state.chat_history)
            memory_export.update({
                'thread_id': st.session_state.get('thread_id', 'default'),
                'memory_settings': {
                    'memory_enabled': st.session_state.get('memory_enabled', False),
                    'memory_type': st.session_state.get('memory_type', 'Short-term'),
                    'max_messages': st.session_state.get('max_messages', 100)
                }
            })
            
            json_str, filename = create_download_data(memory_export, f"memory_export_{st.session_state.get('thread_id', 'default')}")
            st.download_button(
                label="📁 Download",
                data=json_str,
                file_name=filename,
                mime="application/json"
            )
        else:
            st.warning("No chat history to export")


def render_import_memory_action():
    """Render import memory action."""
    uploaded_file = st.file_uploader("📁 Import Memory", type=['json'], key="memory_import_uploader")
    
    if uploaded_file is not None:
        try:
            # Parse the file but don't import yet
            memory_data = json.load(uploaded_file)
            
            # Validate and preview the memory data
            if 'messages' in memory_data:
                messages = memory_data['messages']
                memory_settings = memory_data.get('memory_settings', {})
                format_version = memory_data.get('format_version', 'Unknown')
                
                # Show preview information
                st.info(f"📋 **Memory File Preview:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"• **Messages:** {len(messages)}")
                    st.write(f"• **Thread ID:** {memory_data.get('thread_id', 'default')}")
                with col2:
                    st.write(f"• **Format Version:** {format_version}")
                    st.write(f"• **Memory Type:** {memory_settings.get('memory_type', 'Unknown')}")
                
                # Show first few messages as preview
                if len(messages) > 0:
                    with st.expander("📝 Preview First Few Messages"):
                        for i, msg in enumerate(messages[:3]):
                            role_icon = "👤" if msg.get('role') == 'user' else "🤖"
                            content_preview = str(msg.get('content', ''))[:100] + "..." if len(str(msg.get('content', ''))) > 100 else str(msg.get('content', ''))
                            st.write(f"**{role_icon} {msg.get('role', 'unknown').title()}:** {content_preview}")
                            
                            # Show tool executions if present
                            if msg.get('tool_executions'):
                                st.write(f"   🔧 {len(msg['tool_executions'])} tool execution(s)")
                        
                        if len(messages) > 3:
                            st.write(f"... and {len(messages) - 3} more messages")
                
                # Warning about current chat history
                current_history_count = len(st.session_state.get('chat_history', []))
                if current_history_count > 0:
                    st.warning(f"⚠️ This will replace your current chat history ({current_history_count} messages)")
                
                # Confirmation buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Confirm Import", type="primary", key="confirm_import_btn"):
                        # Actually perform the import
                        st.session_state.chat_history = messages
                        
                        # Apply memory settings if available
                        if memory_settings:
                            if 'memory_enabled' in memory_settings:
                                st.session_state.memory_enabled = memory_settings['memory_enabled']
                            if 'memory_type' in memory_settings:
                                st.session_state.memory_type = memory_settings['memory_type']
                            if 'max_messages' in memory_settings:
                                st.session_state.max_messages = memory_settings['max_messages']
                        
                        # Apply thread ID if available
                        if 'thread_id' in memory_data:
                            st.session_state.thread_id = memory_data['thread_id']
                        
                        st.success(f"✅ Successfully imported {len(messages)} messages!")
                        st.rerun()
                
                with col2:
                    if st.button("❌ Cancel", key="cancel_import_btn"):
                        st.info("Import cancelled")
                        st.rerun()
                        
            else:
                st.error("❌ Invalid memory export file - no 'messages' field found")
                st.info("💡 This app only supports the current enhanced memory format (v3.0+)")
                
        except json.JSONDecodeError:
            st.error("❌ Invalid JSON file")
        except Exception as e:
            st.error(f"❌ Error reading memory file: {str(e)}")
    else:
        st.info("Select a memory export file (.json) to preview and import")


def render_conversation_history_section():
    """Render conversation history viewer section."""
    st.subheader("Conversation History")
    
    if st.session_state.get('chat_history'):
        render_filtered_history()
        render_memory_statistics()
    else:
        st.info("No conversation history available. Start chatting to see memory content!")


def render_history_filters():
    """Render history filter options."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_user = st.checkbox("Show User Messages", value=True, key="memory_filter_user")
    with col2:
        show_assistant = st.checkbox("Show Assistant Messages", value=True, key="memory_filter_assistant")
    with col3:
        show_tools = st.checkbox("Show Tool Executions", value=False, key="memory_filter_tools")
    
    return show_user, show_assistant, show_tools


def render_filtered_history():
    """Render filtered conversation history."""
    show_user, show_assistant, show_tools = render_history_filters()
    
    with st.container():
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user" and show_user:
                with st.expander(f"👤 User Message #{i+1}"):
                    st.write(message["content"])
                    
            elif message["role"] == "assistant" and show_assistant:
                with st.expander(f"🤖 Assistant Message #{i+1}"):
                    st.write(message["content"])
                    if show_tools and message.get("tool_executions"):
                        st.write("**Tool Executions:**")
                        for j, exec_info in enumerate(message["tool_executions"]):
                            st.write(f"• **{exec_info.get('tool_name', 'Unknown')}** at {exec_info.get('timestamp', 'Unknown time')}")
                            if exec_info.get("input"):
                                st.write("Input:")
                                if isinstance(exec_info["input"], dict):
                                    st.json(exec_info["input"], expanded=False)
                                else:
                                    st.code(str(exec_info["input"]), language="text")
                            if exec_info.get("output"):
                                st.write("Output:")
                                if isinstance(exec_info["output"], (dict, list)):
                                    st.json(exec_info["output"], expanded=False)
                                else:
                                    st.code(str(exec_info["output"]), language="text")


def render_memory_statistics():
    """Render memory statistics."""
    st.subheader("Memory Statistics")
    
    stats = calculate_chat_statistics(st.session_state.chat_history)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("User Messages", stats['user_messages'])
        st.metric("Assistant Messages", stats['assistant_messages'])
    with col2:
        st.metric("Tool Executions", stats['tool_executions'])
        st.metric("Messages with Reasoning", stats['messages_with_reasoning'])
    with col3:
        st.metric("Est. Content Tokens", f"{stats['estimated_tokens']:,}")
        st.metric("Est. Thinking Tokens", f"{stats['estimated_thinking_tokens']:,}")
    with col4:
        st.metric("Total Words", f"{stats['total_words']:,}")
        if stats['assistant_messages'] > 0:
            st.metric("Reasoning %", f"{stats['reasoning_percentage']:.1f}%")
        else:
            st.metric("Reasoning %", "0%")


def render_memory_tips():
    """Render memory tips section."""
    with st.expander("💡 Memory Tips"):
        st.markdown("""
        **Memory Types:**
        - **Short-term**: Remembers conversation only within current browser session
        - **Persistent**: Remembers across browser sessions (stores in SQLite database)
        
        **Thread Management:**
        - Use different thread IDs for separate conversation topics
        - Thread IDs help organize conversations by context
        - Switch threads to start fresh conversations while keeping history
        
        **Memory Limits:**
        - Set max messages to control memory usage
        - Higher limits = more context but slower performance
        - Lower limits = faster but may lose conversation context
        
        **Best Practices:**
        - Export important conversations before clearing memory
        - Use descriptive thread IDs (e.g., "project_planning", "debugging_session")
        - Clear memory periodically to maintain performance
        """)


def render_about_tab():
    """Render the about tab."""
    st.image("logo_transparent.png", width=200)
    st.markdown("""
    ### About
    This application demonstrates a Streamlit interface for LangChain MCP (Model Context Protocol) adapters. 
    It allows you to connect to MCP servers and use their tools with different LLM providers.

    For more information, check out:
    - [LangChain MCP Adapters](https://github.com/langchain-ai/langchain-mcp-adapters)
    - [Model Context Protocol](https://modelcontextprotocol.io/introduction)
                
    ### License
    This project is licensed under the MIT License.
    
    ### Acknowledgements
    - This application is built using [Streamlit](https://streamlit.io/) for the frontend.
    
    ### Developer
    - [LinkedIn](https://www.linkedin.com/in/guinacio/)
    - [Github](https://github.com/guinacio)
    """)


def display_tool_executions():
    """Display tool execution history."""
    if st.session_state.tool_executions:
        with st.expander("Tool Execution History", expanded=False):
            for i, exec_record in enumerate(st.session_state.tool_executions):
                st.markdown(f"### Execution #{i+1}: `{exec_record['tool_name']}`")
                
                # Display input properly formatted
                st.write("**Input:**")
                try:
                    if isinstance(exec_record['input'], dict):
                        st.json(exec_record['input'], expanded=False)
                    else:
                        # Try to parse as JSON if it's a string
                        try:
                            import json
                            parsed_input = json.loads(str(exec_record['input']))
                            st.json(parsed_input, expanded=False)
                        except (json.JSONDecodeError, TypeError):
                            # If not valid JSON, display as plain text
                            st.code(str(exec_record['input']), language="text")
                except Exception:
                    # Fallback to text display
                    st.code(str(exec_record['input']), language="text")
                
                # Display output properly formatted
                st.write("**Output:**")
                output = exec_record['output']
                if isinstance(output, (dict, list)):
                    st.json(output, expanded=False)
                else:
                    st.code(str(output), language="text")
                
                st.write(f"**Time:** {exec_record['timestamp']}")
                st.divider()


def render_config_tab():
    """Render the configuration tab for system prompts and model parameters."""
    st.header("🔧 Configuration")
    
    # Initialize config in session state if not exists
    if 'config_system_prompt' not in st.session_state:
        st.session_state.config_system_prompt = DEFAULT_SYSTEM_PROMPT
    if 'config_temperature' not in st.session_state:
        st.session_state.config_temperature = 0.7
    if 'config_max_tokens' not in st.session_state:
        st.session_state.config_max_tokens = None
    if 'config_timeout' not in st.session_state:
        st.session_state.config_timeout = None
    if 'config_use_custom_settings' not in st.session_state:
        st.session_state.config_use_custom_settings = False
    
    # Configuration sections
    render_config_overview()
    st.divider()
    render_system_prompt_section()
    st.divider()
    render_model_parameters_section()
    st.divider()
    render_config_management_section()


def render_config_overview():
    """Render configuration overview."""
    st.subheader("Configuration Overview")
    
    current_provider = st.session_state.get('llm_provider', 'Not Selected')
    current_model = st.session_state.get('selected_model', 'Not Selected')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Provider", current_provider)
    with col2:
        st.metric("Current Model", current_model)
    with col3:
        use_custom = st.session_state.get('config_use_custom_settings', False)
        st.metric("Custom Config", "Enabled" if use_custom else "Disabled")
    
    # Configuration status and apply button
    render_config_status_and_apply()
    
    # Provider capability overview
    if current_provider != 'Not Selected':
        with st.expander("Provider Capabilities"):
            capabilities = get_provider_capabilities(current_provider)
            render_provider_capabilities(capabilities)

def render_config_status_and_apply():
    """Render configuration status and apply button."""
    # Check if configuration has changed
    config_changed = check_config_changed()
    agent_exists = st.session_state.get('agent') is not None
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if agent_exists:
            if st.session_state.get('config_applied', False) and not config_changed:
                st.success("✅ Current configuration is applied to the agent")
            elif config_changed:
                st.warning("⚠️ Configuration changed - click 'Apply Configuration' to update the agent")
            else:
                st.info("ℹ️ Agent is using default configuration")
        else:
            st.info("ℹ️ No agent connected - connect to MCP server or start chat-only mode first")
    
    with col2:
        apply_disabled = not agent_exists or (not config_changed and st.session_state.get('config_applied', False))
        
        if st.button(
            "🔄 Apply Configuration", 
            type="primary",
            disabled=apply_disabled,
            help="Apply current configuration settings to the agent"
        ):
            apply_configuration_to_agent()

def check_config_changed() -> bool:
    """Check if configuration has changed since last application."""
    if not st.session_state.get('config_applied', False):
        return True
    
    # Compare current config with last applied config
    current_config = get_current_config_snapshot()
    last_applied = st.session_state.get('last_applied_config', {})
    
    return current_config != last_applied


def get_current_config_snapshot() -> Dict:
    """Get a snapshot of current configuration for comparison."""
    return {
        'use_custom_settings': st.session_state.get('config_use_custom_settings', False),
        'system_prompt': st.session_state.get('config_system_prompt', ''),
        'temperature': st.session_state.get('config_temperature', 0.7),
        'max_tokens': st.session_state.get('config_max_tokens'),
        'timeout': st.session_state.get('config_timeout'),
        'provider': st.session_state.get('llm_provider', ''),
        'model': st.session_state.get('selected_model', '')
    }


def apply_configuration_to_agent():
    """Apply current configuration to the agent by recreating it."""
    try:
        with st.spinner("Applying configuration to agent..."):
            # Import here to avoid circular imports
            from .ui_components import create_and_configure_agent
            
            # Get current LLM and memory configuration
            llm_config = {
                "provider": st.session_state.get('llm_provider', ''),
                "api_key": st.session_state.get('api_key', ''),  # This might need to be handled differently
                "model": st.session_state.get('selected_model', '')
            }
            
            memory_config = {
                "enabled": st.session_state.get('memory_enabled', False),
                "type": st.session_state.get('memory_type', 'Short-term (Session)'),
                "thread_id": st.session_state.get('thread_id', 'default')
            }
            
            # Get current MCP tools
            mcp_tools = st.session_state.get('tools', [])
            
            # We need to get the API key from the current agent or ask user to reconnect
            # For Ollama, we don't need an API key
            if not llm_config["api_key"] and llm_config["provider"] != "Ollama" and st.session_state.get('agent'):
                st.warning("⚠️ Cannot apply configuration - API key not available. Please reconnect to MCP server or restart chat-only mode with your new configuration.")
                return
            
            # Recreate agent with new configuration
            success = create_and_configure_agent(llm_config, memory_config, mcp_tools)
            
            if success:
                # Mark configuration as applied
                st.session_state.config_applied = True
                st.session_state.last_applied_config = get_current_config_snapshot()
                
                config_summary = []
                if st.session_state.get('config_use_custom_settings', False):
                    config_summary.append("✅ Custom system prompt applied")
                    config_summary.append(f"✅ Temperature: {st.session_state.get('config_temperature', 0.7)}")
                    if st.session_state.get('config_max_tokens'):
                        config_summary.append(f"✅ Max tokens: {st.session_state.get('config_max_tokens')}")
                    if st.session_state.get('config_timeout'):
                        config_summary.append(f"✅ Timeout: {st.session_state.get('config_timeout')}s")
                else:
                    config_summary.append("✅ Default configuration applied")
                
                st.success("🎉 Configuration successfully applied to agent!")
                with st.expander("📋 Applied Settings"):
                    for item in config_summary:
                        st.write(item)
                
                st.rerun()
            else:
                st.error("❌ Failed to apply configuration. Please check your settings and try again.")
                
    except Exception as e:
        st.error(f"❌ Error applying configuration: {str(e)}")
        st.info("💡 Tip: Try reconnecting to your MCP server or restarting chat-only mode to apply the new configuration.")


def get_provider_capabilities(provider: str) -> Dict:
    """Get capabilities for the current provider."""
    from .llm_providers import LLM_PROVIDERS
    
    if provider not in LLM_PROVIDERS:
        return {}
    
    config = LLM_PROVIDERS[provider]
    return {
        "System Prompt Support": "✅" if config.get("supports_system_prompt", False) else "❌",
        "Temperature Range": f"{config.get('temperature_range', (0.0, 1.0))[0]} - {config.get('temperature_range', (0.0, 1.0))[1]}",
        "Max Tokens Range": f"{config.get('max_tokens_range', (1, 4096))[0]:,} - {config.get('max_tokens_range', (1, 4096))[1]:,}",
        "Default Temperature": str(config.get('default_temperature', 0.7)),
        "Default Max Tokens": f"{config.get('default_max_tokens', 4096):,}",
        "Default Timeout": f"{config.get('default_timeout', 600.0)}s",
        "API Key Required": "✅" if config.get("requires_api_key", True) else "❌"
    }


def render_provider_capabilities(capabilities: Dict):
    """Render provider capabilities in a nice format."""
    for key, value in capabilities.items():
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write(f"**{key}:**")
        with col2:
            st.write(value)


def render_system_prompt_section():
    """Render system prompt configuration section."""
    st.subheader("System Prompt Configuration")
    
    current_provider = st.session_state.get('llm_provider', '')
    
    if current_provider and supports_system_prompt(current_provider):
        # Enable/disable custom configuration
        use_custom = st.checkbox(
            "Use Custom Configuration",
            value=st.session_state.get('config_use_custom_settings', False),
            help="Enable to customize system prompt and model parameters",
            key="config_use_custom_checkbox"
        )
        st.session_state.config_use_custom_settings = use_custom
        
        if use_custom:
            # System prompt configuration
            st.markdown("**System Prompt:**")
            system_prompt = st.text_area(
                "Enter your system prompt:",
                value=st.session_state.get('config_system_prompt', DEFAULT_SYSTEM_PROMPT),
                height=200,
                help="This prompt sets the behavior and personality of the AI assistant",
                key="config_system_prompt_textarea"
            )
            st.session_state.config_system_prompt = system_prompt
            
            # Preset system prompts
            with st.expander("📋 System Prompt Presets"):
                render_system_prompt_presets()
        else:
            st.info("Custom configuration is disabled. The agent will use default settings.")
    else:
        if current_provider:
            st.warning(f"⚠️ {current_provider} does not support system prompts")
        else:
            st.info("Please select an LLM provider in the sidebar to configure system prompts")


def render_system_prompt_presets():
    """Render system prompt preset options."""
    presets = {
        "Default Assistant": DEFAULT_SYSTEM_PROMPT,
        "Code Assistant": """You are an expert software developer and coding assistant. You help users with:
- Writing, debugging, and optimizing code
- Explaining complex programming concepts
- Code reviews and best practices
- Architecture and design patterns

When using tools:
- Always explain your approach before executing code
- Provide clear, commented code examples
- Test your solutions when possible
- Suggest improvements and alternatives

Be precise, technical, and thorough in your responses.""",
        
        "Research Assistant": """You are a knowledgeable research assistant. You excel at:
- Finding and analyzing information from various sources
- Synthesizing complex topics into clear explanations
- Fact-checking and source verification
- Academic and professional research

When using tools:
- Always cite your sources
- Provide comprehensive analysis
- Cross-reference multiple sources
- Present balanced perspectives

Be analytical, thorough, and objective in your responses.""",
        
        "Creative Assistant": """You are a creative and imaginative assistant. You specialize in:
- Creative writing and storytelling
- Brainstorming and ideation
- Artistic and design concepts
- Problem-solving with creative approaches

When using tools:
- Think outside the box
- Provide multiple creative alternatives
- Encourage experimentation
- Build on user ideas

Be inspiring, innovative, and supportive in your responses.""",
        
        "Business Assistant": """You are a professional business consultant. You help with:
- Strategic planning and analysis
- Market research and competitive analysis
- Business process optimization
- Financial planning and analysis

When using tools:
- Provide data-driven insights
- Consider multiple stakeholder perspectives
- Focus on practical, actionable recommendations
- Maintain professional standards

Be strategic, analytical, and results-oriented in your responses."""
    }
    
    selected_preset = st.selectbox(
        "Choose a preset:",
        options=list(presets.keys()),
        key="config_preset_selector"
    )
    
    if st.button("Apply Preset", key="config_apply_preset"):
        st.session_state.config_system_prompt = presets[selected_preset]
        st.success(f"Applied '{selected_preset}' preset!")
        st.rerun()


def render_model_parameters_section():
    """Render model parameters configuration section."""
    st.subheader("Model Parameters")
    
    current_provider = st.session_state.get('llm_provider', '')
    use_custom = st.session_state.get('config_use_custom_settings', False)
    
    if current_provider and use_custom:
        col1, col2 = st.columns(2)
        
        with col1:
            render_temperature_config(current_provider)
            render_max_tokens_config(current_provider)
        
        with col2:
            render_timeout_config(current_provider)
            render_parameter_validation()
    else:
        if not current_provider:
            st.info("Please select an LLM provider in the sidebar to configure parameters")
        else:
            st.info("Enable 'Use Custom Configuration' to adjust model parameters")


def render_temperature_config(provider: str):
    """Render temperature configuration."""
    temp_min, temp_max = get_temperature_range(provider)
    default_temp = get_default_temperature(provider)
    
    temperature = st.slider(
        "Temperature",
        min_value=temp_min,
        max_value=temp_max,
        value=st.session_state.get('config_temperature', default_temp),
        step=0.1,
        help=f"Controls randomness. Lower = more focused, Higher = more creative. Range: {temp_min}-{temp_max}",
        key="config_temperature_slider"
    )
    st.session_state.config_temperature = temperature


def render_max_tokens_config(provider: str):
    """Render max tokens configuration."""
    token_min, token_max = get_max_tokens_range(provider)
    default_tokens = get_default_max_tokens(provider)
    
    enable_max_tokens = st.checkbox(
        "Limit Max Tokens",
        value=st.session_state.get('config_max_tokens') is not None,
        help="Limit the maximum number of tokens in the response",
        key="config_enable_max_tokens"
    )
    
    if enable_max_tokens:
        max_tokens = st.number_input(
            "Max Tokens",
            min_value=token_min,
            max_value=token_max,
            value=st.session_state.get('config_max_tokens', default_tokens),
            step=100,
            help=f"Maximum tokens to generate. Range: {token_min:,}-{token_max:,}",
            key="config_max_tokens_input"
        )
        st.session_state.config_max_tokens = max_tokens
    else:
        st.session_state.config_max_tokens = None


def render_timeout_config(provider: str):
    """Render timeout configuration."""
    default_timeout = get_default_timeout(provider)
    
    enable_timeout = st.checkbox(
        "Custom Timeout",
        value=st.session_state.get('config_timeout') is not None,
        help="Set a custom timeout for API requests",
        key="config_enable_timeout"
    )
    
    if enable_timeout:
        timeout = st.number_input(
            "Timeout (seconds)",
            min_value=5.0,
            max_value=300.0,
            value=st.session_state.get('config_timeout', default_timeout),
            step=5.0,
            help="Timeout for API requests in seconds",
            key="config_timeout_input"
        )
        st.session_state.config_timeout = timeout
    else:
        st.session_state.config_timeout = None


def render_parameter_validation():
    """Render parameter validation and preview."""
    current_provider = st.session_state.get('llm_provider', '')
    
    if current_provider:
        temperature = st.session_state.get('config_temperature', 0.7)
        max_tokens = st.session_state.get('config_max_tokens')
        timeout = st.session_state.get('config_timeout')
        
        # Validate parameters
        current_model = st.session_state.get('selected_model', '')
        is_valid, error_msg = validate_model_parameters(
            current_provider, temperature, max_tokens, timeout, current_model
        )
        
        if is_valid:
            st.success("✅ Configuration is valid")
        else:
            st.error(f"❌ {error_msg}")
        
        # Configuration preview
        with st.expander("👀 Configuration Preview"):
            config_preview = {
                "provider": current_provider,
                "model": st.session_state.get('selected_model', 'Not selected'),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": timeout,
                "system_prompt_length": len(st.session_state.get('config_system_prompt', '')) if st.session_state.get('config_system_prompt') else 0
            }
            st.json(config_preview)


def render_config_management_section():
    """Render configuration management section."""
    st.subheader("Configuration Management")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_save_config_action()
    
    with col2:
        render_load_config_action()
    
    with col3:
        render_reset_config_action()
    
    with col4:
        render_export_config_action()
    
    # Debug options
    with st.expander("🔧 Debug Options"):
        debug_system_prompt = st.checkbox(
            "Debug System Prompt",
            value=st.session_state.get('debug_system_prompt', False),
            help="Enable debug logging for system prompt troubleshooting"
        )
        st.session_state.debug_system_prompt = debug_system_prompt
        
        if debug_system_prompt:
            st.info("🔧 Debug mode enabled. System prompt modifier calls will be logged in the chat interface.")
        
        # Additional debug info
        if st.button("📊 Show Debug Info"):
            debug_info = {
                "session_state_keys": list(st.session_state.keys()),
                "agent_exists": st.session_state.get('agent') is not None,
                "config_applied": st.session_state.get('config_applied', False),
                "use_custom_settings": st.session_state.get('config_use_custom_settings', False),
                "system_prompt_length": len(st.session_state.get('config_system_prompt', '')) if st.session_state.get('config_system_prompt') else 0
            }
            st.json(debug_info)


def render_save_config_action():
    """Render save configuration action."""
    if st.button("💾 Save Config", help="Save current configuration"):
        config_name = st.session_state.get('config_name', f"config_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        config_data = {
            'name': config_name,
            'timestamp': datetime.datetime.now().isoformat(),
            'system_prompt': st.session_state.get('config_system_prompt', ''),
            'temperature': st.session_state.get('config_temperature', 0.7),
            'max_tokens': st.session_state.get('config_max_tokens'),
            'timeout': st.session_state.get('config_timeout'),
            'use_custom_settings': st.session_state.get('config_use_custom_settings', False)
        }
        
        # Store in session state (in a real app, you'd save to database)
        if 'saved_configs' not in st.session_state:
            st.session_state.saved_configs = {}
        
        st.session_state.saved_configs[config_name] = config_data
        st.success(f"Configuration saved as '{config_name}'")


def render_load_config_action():
    """Render load configuration action."""
    if 'saved_configs' in st.session_state and st.session_state.saved_configs:
        config_options = list(st.session_state.saved_configs.keys())
        selected_config = st.selectbox(
            "Load Config:",
            options=config_options,
            key="config_load_selector"
        )
        
        if st.button("📂 Load", help="Load selected configuration"):
            if selected_config in st.session_state.saved_configs:
                config_data = st.session_state.saved_configs[selected_config]
                
                st.session_state.config_system_prompt = config_data.get('system_prompt', '')
                st.session_state.config_temperature = config_data.get('temperature', 0.7)
                st.session_state.config_max_tokens = config_data.get('max_tokens')
                st.session_state.config_timeout = config_data.get('timeout')
                st.session_state.config_use_custom_settings = config_data.get('use_custom_settings', False)
                
                st.success(f"Loaded configuration '{selected_config}'")
                st.rerun()
    else:
        st.info("No saved configurations available")


def render_reset_config_action():
    """Render reset configuration action."""
    if st.button("🔄 Reset to Defaults", help="Reset all settings to defaults"):
        st.session_state.config_system_prompt = DEFAULT_SYSTEM_PROMPT
        st.session_state.config_temperature = 0.7
        st.session_state.config_max_tokens = None
        st.session_state.config_timeout = None
        st.session_state.config_use_custom_settings = False
        st.success("Configuration reset to defaults!")
        st.rerun()


def render_export_config_action():
    """Render export configuration action."""
    if st.button("📤 Export Config", help="Export configuration as JSON"):
        config_data = {
            'system_prompt': st.session_state.get('config_system_prompt', ''),
            'temperature': st.session_state.get('config_temperature', 0.7),
            'max_tokens': st.session_state.get('config_max_tokens'),
            'timeout': st.session_state.get('config_timeout'),
            'use_custom_settings': st.session_state.get('config_use_custom_settings', False),
            'exported_at': datetime.datetime.now().isoformat()
        }
        
        json_str, filename = create_download_data(config_data, "agent_config")
        st.download_button(
            label="📁 Download Config",
            data=json_str,
            file_name=filename,
            mime="application/json"
        )