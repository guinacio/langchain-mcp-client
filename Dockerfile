FROM python:3.12-slim AS builder

WORKDIR /app

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files and readme used by build backend
COPY pyproject.toml uv.lock README.md ./

# Sync dependencies (no dev)
RUN uv sync --frozen --no-dev

FROM python:3.12-slim
WORKDIR /app

# Copy virtualenv from builder (keep same path)
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy application source
COPY . .

# Expose ports for Streamlit app and MCP server
EXPOSE 8501
EXPOSE 8000

# Default command: start MCP server and Streamlit (use explicit venv binaries)
CMD ["sh", "-lc", "/app/.venv/bin/python weather_server.py & /app/.venv/bin/streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]