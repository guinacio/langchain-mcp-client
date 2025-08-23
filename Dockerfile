FROM python:3.12-slim AS builder

WORKDIR /app

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Sync dependencies (no dev)
RUN uv sync --frozen --no-dev

FROM python:3.12-slim
WORKDIR /app

# Copy virtualenv from builder
COPY --from=builder /.venv /.venv
ENV PATH="/.venv/bin:$PATH"

# Copy application source
COPY . .

# Expose ports for Streamlit app and MCP server
EXPOSE 8501
EXPOSE 8000

# Default command: start MCP server and Streamlit
CMD ["bash", "-lc", "python weather_server.py & streamlit run app.py --server.port=8501"]