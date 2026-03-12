# Use a slim Python image
FROM python:3.10-slim AS base

# Set build argument for GitHub Access Key (Passed at build time)

# Set a dedicated working directory (not root)
WORKDIR /usr/src/app

### 🔹 SUBMODULE HANDLING (Using HTTPS with Token)

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*


### 🔹 MAIN BUILD

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential  \
    libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy only requirements first for caching dependencies
COPY r.txt ./

# Install dependencies with no cache
RUN pip install -r r.txt

# Copy the rest of the project files into the working directory
COPY . .

# Set environment variables
ENV PYTHONPATH=/usr/src/app:$PYTHONPATH
ENV PYTHONUNBUFFERED=1 \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8000 \
    MCP_PATH=/mcp \
    MCP_JSON_RESPONSE=true \
    MCP_STATELESS_HTTP=true \
    LOCAL_PATH=/usr/src/app/local_data \
    PORT=8000 \
    GCS_MOUNT_PATH=/mnt/bucket/

# Create local in-container path for mounted data/artifacts
RUN mkdir -p "${LOCAL_PATH}"

# Copy the startup script
COPY startup.sh .

# Make the script executable
RUN chmod +x startup.sh

# Expose the application port
EXPOSE 8000

# Local volume path inside the container
VOLUME ["/usr/src/app/local_data"]

# Start FastMCP server
CMD ["py -m mcp_server.server"]