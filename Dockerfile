FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/app
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    wget \
    curl \
    nginx \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN useradd -m -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Build C++ components (if CMake is available)
RUN if command -v cmake >/dev/null 2>&1; then \
        mkdir -p build && cd build && \
        cmake .. && make; \
    else \
        echo "CMake not available, skipping C++ build"; \
    fi

# Set up web server configuration
COPY docker/nginx.conf /etc/nginx/sites-available/default
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Create necessary directories
RUN mkdir -p /var/log/supervisor
RUN mkdir -p /app/logs

# Set permissions
RUN chown -R appuser:appuser /app
RUN chmod +x scripts/*.py

# Create startup script
RUN echo '#!/bin/bash\n\
echo "Starting TrafficFlowOpt..."\n\
echo "Fetching real traffic data..."\n\
python3 scripts/fetch_real_data.py\n\
echo "Generating web assets..."\n\
python3 scripts/generate_web_assets.py\n\
echo "Starting main application..."\n\
python3 src/python/main.py\n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose ports
EXPOSE 80 8050

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost/ || exit 1

# Default command
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]