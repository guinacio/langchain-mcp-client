"""
Memory and conversation history tools.

This module provides tools for managing conversation history
and memory functionality within the agent system.
"""

import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
import streamlit as st


class HistoryFilter(BaseModel):
    """Input schema for the conversation history tool."""
    message_type: Optional[str] = Field(
        default="all", 
        description="Filter by message type: 'user', 'assistant', 'tool', or 'all'"
    )
    last_n_messages: Optional[int] = Field(
        default=10, 
        description="Number of recent messages to retrieve (max 100)"
    )
    search_query: Optional[str] = Field(
        default=None, 
        description="Search for messages containing this text. Supports: simple text, boolean operators (AND, OR, NOT), and regex patterns (prefix with 'regex:')"
    )
    sort_order: Optional[str] = Field(
        default="newest_first",
        description="Sort order: 'newest_first' or 'oldest_first'"
    )
    date_from: Optional[str] = Field(
        default=None,
        description="Filter messages from this date (YYYY-MM-DD format)"
    )
    date_to: Optional[str] = Field(
        default=None,
        description="Filter messages to this date (YYYY-MM-DD format)"
    )
    include_metadata: Optional[bool] = Field(
        default=True,
        description="Include timestamps and message IDs in the response"
    )


class ConversationHistoryTool(BaseTool):
    """Custom tool for retrieving conversation history from the current session."""
    
    name: str = "get_conversation_history"
    description: str = """
    Retrieve conversation history from the current session with advanced filtering and search options.
    
    This tool allows you to access previous messages in the conversation to:
    - Reference earlier discussions with timestamps and message IDs
    - Summarize conversation topics with chronological context
    - Find specific information mentioned before with date filtering
    - Maintain context across interactions with sorting options
    - Search through conversation history with powerful query capabilities
    
    Enhanced features:
    - Timestamps for each message
    - Unique message IDs for precise referencing
    - Date range filtering
    - Flexible sorting (newest/oldest first)
    - Rich metadata including tool execution details
    
    Advanced Search Capabilities:
    - Simple text search: "weather"
    - Boolean operators: "weather AND temperature", "sunny OR cloudy", "weather NOT rain"
    - Regex patterns: "regex:\\d{2}°[CF]" (finds temperature patterns like "72°F")
    - Complex queries: "(weather OR climate) AND NOT rain"
    - Tool-specific search: searches through tool names, inputs, and outputs
    
    Search Examples:
    - "weather AND San Francisco" - Messages containing both terms
    - "temperature OR humidity OR pressure" - Messages with any weather metric
    - "NOT error" - Messages that don't contain "error"
    - "regex:https?://\\S+" - Find messages with URLs
    - "regex:\\$\\d+(\\.\\d{2})?" - Find messages with dollar amounts
    """
    args_schema: type[BaseModel] = HistoryFilter
    
    def _run(
        self,
        message_type: str = "all",
        last_n_messages: int = 10,
        search_query: Optional[str] = None,
        sort_order: str = "newest_first",
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        include_metadata: bool = True
    ) -> str:
        """Synchronous implementation."""
        return self._get_history(
            message_type, last_n_messages, search_query, 
            sort_order, date_from, date_to, include_metadata
        )
    
    async def _arun(
        self,
        message_type: str = "all",
        last_n_messages: int = 10,
        search_query: Optional[str] = None,
        sort_order: str = "newest_first",
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        include_metadata: bool = True
    ) -> str:
        """Asynchronous implementation."""
        return self._get_history(
            message_type, last_n_messages, search_query,
            sort_order, date_from, date_to, include_metadata
        )
    
    def _get_history(
        self,
        message_type: str = "all",
        last_n_messages: int = 10,
        search_query: Optional[str] = None,
        sort_order: str = "newest_first",
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        include_metadata: bool = True
    ) -> str:
        """Get conversation history from session state with enhanced filtering."""
        try:
            # Access session state through streamlit
            if not hasattr(st.session_state, 'chat_history') or not st.session_state.chat_history:
                return "No conversation history available yet."
            
            # Limit the number of messages - ensure we have valid integers
            last_n_messages = int(last_n_messages) if last_n_messages is not None else 10
            last_n_messages = min(max(1, last_n_messages), 100)  # Increased max to 100
            
            # Start with all messages and add metadata if missing
            all_messages = self._ensure_message_metadata(st.session_state.chat_history)
            
            # Apply date filtering first if provided
            if date_from or date_to:
                all_messages = self._filter_by_date_range(all_messages, date_from, date_to)
            
            # Filter by message type
            filtered_messages = []
            if message_type != "all":
                if message_type == "tool":
                    # Show messages that have tool executions
                    for msg in all_messages:
                        if (msg.get("role") == "assistant" and 
                            msg.get("tool_executions") and len(msg.get("tool_executions", [])) > 0):
                            filtered_messages.append(msg)
                else:
                    for msg in all_messages:
                        if msg.get("role") == message_type:
                            filtered_messages.append(msg)
            else:
                filtered_messages = all_messages
            
            # Apply search filter if provided
            if search_query and search_query.strip():
                filtered_messages = self._filter_by_search(filtered_messages, search_query)
            
            # Apply sorting
            if sort_order == "oldest_first":
                filtered_messages = sorted(filtered_messages, key=lambda x: x.get("timestamp", ""))
            else:  # newest_first (default)
                filtered_messages = sorted(filtered_messages, key=lambda x: x.get("timestamp", ""), reverse=True)
            
            # Take the requested number of messages
            recent_filtered_messages = filtered_messages[:last_n_messages] if filtered_messages else []
            
            if not recent_filtered_messages:
                return f"No messages found matching your criteria (type: {message_type}, search: '{search_query or 'none'}', date range: {date_from or 'any'} to {date_to or 'any'})"
            
            # Format the history in a clean, readable way
            return self._format_history_response(recent_filtered_messages, include_metadata, sort_order)
            
        except Exception as e:
            return f"Error retrieving conversation history: {str(e)}"
    
    def _ensure_message_metadata(self, messages: list) -> list:
        """Ensure all messages have timestamps and IDs."""
        enhanced_messages = []
        current_time = datetime.datetime.now()
        
        for i, msg in enumerate(messages):
            enhanced_msg = msg.copy()
            
            # Add message ID if missing
            if "message_id" not in enhanced_msg:
                enhanced_msg["message_id"] = f"msg_{i+1:04d}"
            
            # Add timestamp if missing (estimate based on position)
            if "timestamp" not in enhanced_msg:
                # Estimate timestamp by going back in time for older messages
                estimated_time = current_time - datetime.timedelta(minutes=(len(messages) - i - 1) * 2)
                enhanced_msg["timestamp"] = estimated_time.strftime('%Y-%m-%d %H:%M:%S')
            
            enhanced_messages.append(enhanced_msg)
        
        return enhanced_messages
    
    def _filter_by_date_range(self, messages: list, date_from: Optional[str], date_to: Optional[str]) -> list:
        """Filter messages by date range."""
        if not date_from and not date_to:
            return messages
        
        filtered = []
        for msg in messages:
            msg_timestamp = msg.get("timestamp", "")
            if not msg_timestamp:
                continue
            
            try:
                # Extract date part from timestamp
                msg_date = msg_timestamp.split(' ')[0]  # Get YYYY-MM-DD part
                
                # Check date range
                if date_from and msg_date < date_from:
                    continue
                if date_to and msg_date > date_to:
                    continue
                
                filtered.append(msg)
            except (ValueError, IndexError):
                # If timestamp parsing fails, include the message
                filtered.append(msg)
        
        return filtered
    
    def _filter_by_search(self, messages: list, search_query: str) -> list:
        """Filter messages by search query with support for boolean operators and regex."""
        import re
        
        search_filtered = []
        search_term = search_query.strip()
        
        # Check if it's a regex search
        if search_term.lower().startswith('regex:'):
            return self._filter_by_regex(messages, search_term[6:].strip())
        
        # Check if it contains boolean operators
        if any(op in search_term.upper() for op in [' AND ', ' OR ', ' NOT ']):
            return self._filter_by_boolean_search(messages, search_term)
        
        # Simple text search (original functionality)
        search_term_lower = search_term.lower()
        for msg in messages:
            if self._message_contains_text(msg, search_term_lower):
                search_filtered.append(msg)
        
        return search_filtered
    
    def _filter_by_regex(self, messages: list, regex_pattern: str) -> list:
        """Filter messages using regex pattern."""
        import re
        
        try:
            compiled_pattern = re.compile(regex_pattern, re.IGNORECASE | re.MULTILINE)
        except re.error as e:
            # If regex is invalid, fall back to simple text search
            return self._filter_by_search(messages, regex_pattern)
        
        filtered = []
        for msg in messages:
            content = self._get_message_searchable_content(msg)
            if compiled_pattern.search(content):
                filtered.append(msg)
        
        return filtered
    
    def _filter_by_boolean_search(self, messages: list, search_query: str) -> list:
        """Filter messages using boolean operators (AND, OR, NOT)."""
        import re
        
        # Parse the boolean query
        # This is a simplified parser - for production, you might want a more robust one
        query = search_query.upper()
        
        # Split by OR first (lowest precedence)
        or_parts = re.split(r'\s+OR\s+', query)
        
        filtered = []
        for msg in messages:
            message_matches = False
            
            # Check each OR part
            for or_part in or_parts:
                and_parts = re.split(r'\s+AND\s+', or_part.strip())
                part_matches = True
                
                # All AND parts must match
                for and_part in and_parts:
                    and_part = and_part.strip()
                    
                    # Handle NOT operator
                    if and_part.startswith('NOT '):
                        term = and_part[4:].strip().strip('"\'').lower()
                        if self._message_contains_text(msg, term):
                            part_matches = False
                            break
                    else:
                        # Remove quotes if present
                        term = and_part.strip('"\'').lower()
                        if not self._message_contains_text(msg, term):
                            part_matches = False
                            break
                
                if part_matches:
                    message_matches = True
                    break
            
            if message_matches:
                filtered.append(msg)
        
        return filtered
    
    def _message_contains_text(self, msg: dict, search_term: str) -> bool:
        """Check if a message contains the search term."""
        content = self._get_message_searchable_content(msg).lower()
        return search_term in content
    
    def _get_message_searchable_content(self, msg: dict) -> str:
        """Get all searchable content from a message."""
        content_parts = []
        
        # Main message content
        content_parts.append(str(msg.get("content", "")))
        
        # Tool executions content
        if msg.get("tool_executions"):
            for exec_info in msg.get("tool_executions", []):
                content_parts.append(str(exec_info.get("tool_name", "")))
                content_parts.append(str(exec_info.get("input", "")))
                content_parts.append(str(exec_info.get("output", "")))
        
        return " ".join(content_parts)
    
    def _format_history_response(self, messages: list, include_metadata: bool, sort_order: str) -> str:
        """Format the history response with enhanced information."""
        formatted_history = []
        
        # Header with summary
        total_messages = len(messages)
        date_range = self._get_date_range(messages)
        
        formatted_history.append(f":material/assignment: Conversation history ({total_messages} message{'s' if total_messages != 1 else ''})")
        if date_range:
            formatted_history.append(f":material/date_range: Date range: {date_range}")
        formatted_history.append(f":material/sync: Sort order: {sort_order.replace('_', ' ').title()}")
        formatted_history.append("")
        
        # Messages
        for i, msg in enumerate(messages, 1):
            role = msg.get("role", "unknown")
            content = str(msg.get("content", "")).strip()
            message_id = msg.get("message_id", f"msg_{i}")
            timestamp = msg.get("timestamp", "Unknown time")
            
            # Format message header
            if include_metadata:
                header = f"{i}. [{message_id}] {role.title()} ({timestamp}):"
            else:
                header = f"{i}. {role.title()}:"
            
            formatted_history.append(header)
            formatted_history.append(f"   {content}")
            
            # Add tool execution information if available
            if msg.get("tool_executions"):
                formatted_history.append("   :material/handyman: Tools used:")
                for tool_exec in msg.get("tool_executions", []):
                    tool_name = tool_exec.get("tool_name", "Unknown")
                    tool_time = tool_exec.get("timestamp", "Unknown time")
                    formatted_history.append(f"      - {tool_name} (executed at {tool_time})")
                    
                    # Show brief input/output if available
                    if tool_exec.get("input"):
                        input_str = str(tool_exec["input"])
                        if len(input_str) > 100:
                            input_str = input_str[:100] + "..."
                        formatted_history.append(f"        Input: {input_str}")
                    
                    if tool_exec.get("output"):
                        output_str = str(tool_exec["output"])
                        if len(output_str) > 150:
                            output_str = output_str[:150] + "..."
                        formatted_history.append(f"        Output: {output_str}")
            
            formatted_history.append("")  # Add spacing between messages
        
        return "\n".join(formatted_history)
    
    def _get_date_range(self, messages: list) -> str:
        """Get the date range of messages."""
        if not messages:
            return ""
        
        timestamps = [msg.get("timestamp", "") for msg in messages if msg.get("timestamp")]
        if not timestamps:
            return ""
        
        try:
            dates = [ts.split(' ')[0] for ts in timestamps]  # Extract date parts
            min_date = min(dates)
            max_date = max(dates)
            
            if min_date == max_date:
                return min_date
            else:
                return f"{min_date} to {max_date}"
        except (ValueError, IndexError):
            return "Various dates"


def create_history_tool() -> ConversationHistoryTool:
    """Create an instance of the conversation history tool."""
    return ConversationHistoryTool()


def format_chat_history_for_export(chat_history: List[Dict]) -> Dict:
    """
    Format chat history for export with comprehensive metadata.
    
    Args:
        chat_history: List of message dictionaries
        
    Returns:
        Formatted export data with enhanced structure
    """
    export_data = {
        'format_version': '3.1',  # Increment version to include thinking content
        'export_timestamp': datetime.datetime.now().isoformat(),
        'total_messages': len(chat_history),
        'messages': [],
        'statistics': calculate_chat_statistics(chat_history)
    }
    
    for message in chat_history:
        formatted_message = {
            'role': message.get('role', 'unknown'),
            'content': message.get('content', ''),
            'timestamp': message.get('timestamp', ''),
            'message_id': message.get('message_id', ''),
            'word_count': len(message.get('content', '').split())
        }
        
        # Add model information for assistant messages
        if message.get('role') == 'assistant':
            if 'model_provider' in message:
                formatted_message['model_provider'] = message['model_provider']
            if 'model_name' in message:
                formatted_message['model_name'] = message['model_name']
            if 'response_time' in message:
                formatted_message['response_time'] = message['response_time']
            
            # Add thinking content if available
            if 'thinking' in message and message['thinking']:
                formatted_message['thinking'] = message['thinking']
                formatted_message['has_reasoning'] = True
                formatted_message['thinking_word_count'] = len(message['thinking'].split())
            else:
                formatted_message['has_reasoning'] = False
        
        # Add tool execution information if available
        if 'tool_executions' in message and message['tool_executions']:
            formatted_message['tool_executions'] = message['tool_executions']
            formatted_message['tool_count'] = len(message['tool_executions'])
        else:
            formatted_message['tool_count'] = 0
        
        export_data['messages'].append(formatted_message)
    
    return export_data


def calculate_chat_statistics(chat_history: list) -> dict:
    """
    Calculate comprehensive statistics about the chat history.
    
    Args:
        chat_history: List of chat messages
        
    Returns:
        Dictionary containing various statistics
    """
    total_messages = len(chat_history)
    user_messages = len([msg for msg in chat_history if msg.get('role') == 'user'])
    assistant_messages = len([msg for msg in chat_history if msg.get('role') == 'assistant'])
    
    # Count tool executions across all messages
    tool_executions = 0
    for msg in chat_history:
        if 'tool_executions' in msg and msg['tool_executions']:
            tool_executions += len(msg['tool_executions'])
    
    # Estimate token count (rough approximation: 1 token ≈ 0.75 words)
    total_words = 0
    thinking_words = 0
    messages_with_reasoning = 0
    
    for msg in chat_history:
        content = msg.get('content', '')
        if content:
            total_words += len(content.split())
        
        # Count thinking words and messages with reasoning
        if msg.get('role') == 'assistant' and 'thinking' in msg and msg['thinking']:
            thinking_content = msg['thinking']
            thinking_words += len(thinking_content.split())
            messages_with_reasoning += 1
    
    estimated_tokens = int(total_words * 1.33)  # Slightly higher ratio for more accurate estimate
    estimated_thinking_tokens = int(thinking_words * 1.33)
    
    return {
        'total_messages': total_messages,
        'user_messages': user_messages,
        'assistant_messages': assistant_messages,
        'tool_executions': tool_executions,
        'total_words': total_words,
        'thinking_words': thinking_words,
        'messages_with_reasoning': messages_with_reasoning,
        'estimated_tokens': estimated_tokens,
        'estimated_thinking_tokens': estimated_thinking_tokens,
        'reasoning_percentage': (messages_with_reasoning / assistant_messages * 100) if assistant_messages > 0 else 0
    } 