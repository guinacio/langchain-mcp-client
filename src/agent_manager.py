"""
Agent management and execution functionality.

This module handles the creation, configuration, and execution
of LangGraph agents with various tools and memory configurations.
"""

from typing import Dict, List, Optional
import streamlit as st
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, AIMessage

from .memory_tools import create_history_tool
from .database import PersistentStorageManager
from .utils import model_supports_tools


async def run_agent(agent, message_content) -> Dict:
    """Run the agent with the provided (possibly multimodal) message content."""
    return await agent.ainvoke({"messages": [HumanMessage(content=message_content)]})


async def stream_agent_response(agent, message_content, config: Dict = None):
    """Stream agent response with real-time updates."""
    messages = [HumanMessage(content=message_content)]
    
    if config:
        async for event in agent.astream({"messages": messages}, config):
            yield event
    else:
        async for event in agent.astream({"messages": messages}):
            yield event


async def stream_agent_events(agent, message_content, config: Dict = None):
    """Stream agent events for more detailed streaming control."""
    messages = [HumanMessage(content=message_content)]
    
    if config:
        async for event in agent.astream_events({"messages": messages}, config, version="v2"):
            yield event
    else:
        async for event in agent.astream_events({"messages": messages}, version="v2"):
            yield event


async def run_tool(tool, **kwargs):
    """Run a tool with the provided parameters."""
    return await tool.ainvoke(kwargs)





def create_agent_with_tools(
    llm,
    mcp_tools: List[BaseTool],
    memory_enabled: bool = False,
    memory_type: str = "Short-term (Session)",
    persistent_storage: Optional[PersistentStorageManager] = None
):
    """
    Create a LangGraph agent with the specified configuration.
    
    Args:
        llm: The language model to use
        mcp_tools: List of MCP tools
        memory_enabled: Whether to enable memory
        memory_type: Type of memory to use
        persistent_storage: PersistentStorageManager instance for persistent memory
    
    Returns:
        Configured agent and checkpointer
    """
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import MessagesState
    from langchain_core.messages import BaseMessage
    
    # Check if the model supports tools
    model_name = getattr(llm, 'model_name', getattr(llm, 'model', ''))
    supports_tools = model_supports_tools(model_name)
    
    # Additionally, try to test tool binding capability
    if supports_tools:
        try:
            # Try to bind an empty list of tools to see if the model actually supports tool calling
            test_model = llm.bind_tools([])
        except Exception:
            supports_tools = False
    
    # Start with MCP tools only if model supports tools
    agent_tools = []
    checkpointer = None
    
    # Set up memory functionality if enabled
    if memory_enabled:
        if memory_type == "Persistent (Cross-session)" and persistent_storage:
            # Use SQLite checkpointer for persistent storage
            checkpointer = persistent_storage.get_checkpointer_sync()
        else:
            # Use in-memory checkpointer for short-term storage
            checkpointer = InMemorySaver()
    
    # Only add tools if the model supports them
    if supports_tools:
        agent_tools = mcp_tools.copy()
        
        # Add history tool when memory is enabled and model supports tools
        if memory_enabled:
            agent_tools.append(create_history_tool())
        
        # Check if LLM has a system prompt
        system_prompt = getattr(llm, '_system_prompt', None)
        
        # Create the agent with optional system prompt
        if system_prompt:
            # Use the prompt parameter to add system prompt
            agent = create_react_agent(
                llm, 
                agent_tools, 
                checkpointer=checkpointer, 
                prompt=system_prompt
            )
        else:
            agent = create_react_agent(llm, agent_tools, checkpointer=checkpointer)
    else:
        # Create a simple conversational agent without tools for models that don't support them
        def call_model(state: MessagesState):
            # Get system prompt if available
            system_prompt = getattr(llm, '_system_prompt', None)
            messages = state["messages"]
            
            if system_prompt and (not messages or not any(msg.type == "system" for msg in messages)):
                from langchain_core.messages import SystemMessage
                messages = [SystemMessage(content=system_prompt)] + messages
            
            response = llm.invoke(messages)
            return {"messages": [response]}
        
        # Build a simple StateGraph for non-tool models
        builder = StateGraph(MessagesState)
        builder.add_node("call_model", call_model)
        builder.add_edge(START, "call_model")
        builder.add_edge("call_model", END)
        
        # Compile with checkpointer if memory is enabled
        agent = builder.compile(checkpointer=checkpointer)
    
    return agent, checkpointer


def get_agent_config_summary(
    provider: str,
    model: str,
    mcp_tool_count: int,
    memory_enabled: bool,
    memory_type: str = None,
    thread_id: str = None
) -> Dict[str, str]:
    """
    Generate a summary of the agent configuration.
    
    Args:
        provider: LLM provider name
        model: Model name
        mcp_tool_count: Number of MCP tools
        memory_enabled: Whether memory is enabled
        memory_type: Type of memory if enabled
        thread_id: Thread ID if memory is enabled
    
    Returns:
        Dictionary with configuration summary
    """
    config = {
        "provider": provider,
        "model": model,
        "mcp_tools": str(mcp_tool_count),
        "memory": "Enabled" if memory_enabled else "Disabled"
    }
    
    if memory_enabled:
        config["memory_type"] = memory_type or "Short-term"
        config["thread_id"] = thread_id or "default"
        config["total_tools"] = str(mcp_tool_count + 1)  # +1 for history tool
    else:
        config["total_tools"] = str(mcp_tool_count)
    
    return config


def validate_agent_configuration(
    llm_provider: str,
    api_key: str,
    model_name: str,
    memory_enabled: bool = False,
    memory_type: str = None,
    thread_id: str = None
) -> tuple[bool, str]:
    """
    Validate agent configuration before creation.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Import here to avoid circular imports
    from .llm_providers import validate_provider_config
    from .utils import is_valid_thread_id
    
    # Validate LLM configuration
    is_valid, error = validate_provider_config(llm_provider, api_key, model_name)
    if not is_valid:
        return False, error
    
    # Validate memory configuration
    if memory_enabled:
        if memory_type not in ["Short-term (Session)", "Persistent (Cross-session)"]:
            return False, "Invalid memory type"
        
        if thread_id and not is_valid_thread_id(thread_id):
            return False, "Invalid thread ID format"
    
    return True, ""


def prepare_agent_invocation_config(
    memory_enabled: bool,
    thread_id: str = "default"
) -> Dict:
    """
    Prepare configuration for agent invocation.
    
    Args:
        memory_enabled: Whether memory is enabled
        thread_id: Thread ID for memory
    
    Returns:
        Configuration dictionary for agent invocation
    """
    if memory_enabled:
        return {"configurable": {"thread_id": thread_id}}
    return {}


def extract_tool_executions_from_response(response: Dict) -> List[Dict]:
    """
    Extract tool execution information from agent response.
    
    Args:
        response: Agent response dictionary
    
    Returns:
        List of tool execution records for the CURRENT interaction only
    """
    from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
    import datetime
    
    tool_executions = []
    current_timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if "messages" not in response:
        return tool_executions
    
    messages = response["messages"]
    
    # When memory is enabled, we need to find only the NEW messages from this interaction
    # Strategy: Look for the last HumanMessage (current user input) and then get 
    # any AIMessage and ToolMessage that come after it
    
    # Find the index of the last HumanMessage (current interaction)
    last_human_idx = -1
    for i in reversed(range(len(messages))):
        if isinstance(messages[i], HumanMessage):
            last_human_idx = i
            break
    
    if last_human_idx == -1:
        # No human message found, fall back to looking at the last few messages
        # This handles the case where memory is disabled
        recent_messages = messages[-10:]  # Look at last 10 messages max
    else:
        # Get messages after the last human message (these are from current interaction)
        recent_messages = messages[last_human_idx + 1:]
    
    # Find AI messages with tool calls in the recent messages
    for msg in recent_messages:
        if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
            # Process tool calls from this AI message
            for tool_call in msg.tool_calls:
                # Find corresponding ToolMessage in recent messages
                tool_output = next(
                    (m.content for m in recent_messages
                     if isinstance(m, ToolMessage) and 
                     hasattr(m, 'tool_call_id') and
                     m.tool_call_id == tool_call['id']),
                    None
                )
                
                if tool_output:
                    tool_executions.append({
                        "tool_name": tool_call['name'],
                        "input": tool_call['args'],
                        "output": tool_output,
                        "timestamp": current_timestamp
                    })
    
    return tool_executions


def extract_assistant_response(response: Dict) -> str:
    """
    Extract the assistant's text response from the agent response.
    
    Args:
        response: Agent response dictionary
    
    Returns:
        Assistant's text response
    """
    from langchain_core.messages import HumanMessage, AIMessage
    from .utils import coerce_content_to_text
    
    if "messages" not in response:
        return ""
    
    # When memory is enabled, the response contains the entire conversation history
    # We need to find the LAST (most recent) AIMessage that has actual content
    ai_messages = []
    for msg in response["messages"]:
        if isinstance(msg, AIMessage) and hasattr(msg, "content") and msg.content:
            content = coerce_content_to_text(msg.content).strip()
            # Skip tool call artifacts and empty content
            if (content and 
                content != "<|tool_call|>[]" and 
                not content.startswith('<|tool_call|>') and
                not content.endswith('[]')):
                ai_messages.append(content)
    
    # Return the last (most recent) AI message content
    if ai_messages:
        return ai_messages[-1]
    
    return ""


async def sync_imported_messages_to_checkpointer(imported_messages: List[Dict], thread_id: str = "default") -> bool:
    """
    Sync imported chat history messages to the LangGraph checkpointer.
    
    This function takes imported messages and runs them through the agent
    to populate the checkpointer with the conversation history.
    
    Args:
        imported_messages: List of message dictionaries from imported memory
        thread_id: Thread ID to use for the checkpointer
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not hasattr(st.session_state, 'agent') or not st.session_state.agent:
            return False
        
        if not hasattr(st.session_state, 'checkpointer') or not st.session_state.checkpointer:
            return False
        
        agent = st.session_state.agent
        config = {"configurable": {"thread_id": thread_id}}
        
        # Process messages in pairs (user message followed by assistant response)
        user_message = None
        
        for msg in imported_messages:
            role = msg.get('role', '').lower()
            content = msg.get('content', '')
            
            if not content:
                continue
                
            if role == 'user':
                user_message = content
                
            elif role == 'assistant' and user_message:
                # We have a user-assistant pair, simulate the conversation
                # by running the agent but bypassing the actual LLM call
                try:
                    # Create the message objects
                    messages = [HumanMessage(content=user_message), AIMessage(content=content)]
                    
                    # Manually update the checkpointer state
                    # This approach directly saves the conversation state to the checkpointer
                    if hasattr(st.session_state.checkpointer, 'put'):
                        # Build the checkpoint state that matches LangGraph's expected format
                        from uuid import uuid4
                        checkpoint_id = str(uuid4())
                        
                        # Create a checkpoint tuple that matches LangGraph's format
                        checkpoint_state = {
                            "messages": messages
                        }
                        
                        # Save to checkpointer
                        st.session_state.checkpointer.put(
                            config=config,
                            checkpoint={
                                "v": 1,
                                "ts": checkpoint_id,
                                "id": checkpoint_id,
                                "channel_values": checkpoint_state,
                                "pending_sends": []
                            },
                            metadata={"source": "imported", "step": -1, "writes": {}}
                        )
                    
                    user_message = None  # Reset for next pair
                    
                except Exception as e:
                    # Log error but continue with other messages
                    print(f"Error processing message pair: {str(e)}")
                    user_message = None
                    continue
        
        return True
        
    except Exception as e:
        print(f"Error syncing imported messages to checkpointer: {str(e)}")
        return False


def sync_imported_messages_to_checkpointer_sync(imported_messages: List[Dict], thread_id: str = "default") -> bool:
    """
    Synchronous wrapper for sync_imported_messages_to_checkpointer.
    
    Args:
        imported_messages: List of message dictionaries from imported memory
        thread_id: Thread ID to use for the checkpointer
    
    Returns:
        bool: True if successful, False otherwise
    """
    from .utils import run_async
    
    try:
        return run_async(sync_imported_messages_to_checkpointer(imported_messages, thread_id))
    except Exception as e:
        print(f"Error in sync wrapper: {str(e)}")
        return False


def simple_sync_imported_messages_to_checkpointer(imported_messages: List[Dict], thread_id: str = "default") -> bool:
    """
    Simplified approach to sync imported messages to checkpointer.
    
    This uses a simpler method by running actual agent calls with the imported messages
    to ensure the checkpointer is properly populated.
    
    Args:
        imported_messages: List of message dictionaries from imported memory  
        thread_id: Thread ID to use for the checkpointer
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not hasattr(st.session_state, 'agent') or not st.session_state.agent:
            return False
        
        agent = st.session_state.agent
        config = {"configurable": {"thread_id": thread_id}}
        
        # Create LangChain message objects from imported messages
        langchain_messages = []
        for msg in imported_messages:
            role = msg.get('role', '').lower()
            content = msg.get('content', '')
            
            if not content:
                continue
                
            if role == 'user':
                langchain_messages.append(HumanMessage(content=content))
            elif role == 'assistant':
                langchain_messages.append(AIMessage(content=content))
        
        # If we have messages, invoke the agent with the full conversation
        # This will populate the checkpointer with the complete history
        if langchain_messages:
            # Run the agent with all the imported messages at once
            # This should trigger the checkpointer to save the conversation state
            from .utils import run_async
            result = run_async(agent.ainvoke({"messages": langchain_messages}, config))
            return True
        
        return False
        
    except Exception as e:
        print(f"Error in simple sync: {str(e)}")
        return False 