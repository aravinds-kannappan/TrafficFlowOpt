# CMakeLists.txt - Build system for TrafficFlowOpt
cmake_minimum_required(VERSION 3.15)
project(TrafficFlowOpt VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()

# Find required packages
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)

# Optional packages for enhanced functionality
find_package(OpenMP)
find_package(Boost COMPONENTS system filesystem thread)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)

# Source files
set(SOURCES
    src/TrafficFlowOpt.cpp
    src/SensorAPI.cpp
    src/LinearAlgebraModule.cpp
    src/CalculusModule.cpp
    src/GraphTheoryModule.cpp
    src/JAXIntegration.cpp
    src/OptimizationEngine.cpp
    src/DataProcessor.cpp
    src/NetworkManager.cpp
    src/ConfigManager.cpp
)

# Header files
set(HEADERS
    include/TrafficFlowOpt.h
    include/SensorAPI.h
    include/LinearAlgebraModule.h
    include/CalculusModule.h
    include/GraphTheoryModule.h
    include/JAXIntegration.h
    include/OptimizationEngine.h
    include/DataProcessor.h
    include/NetworkManager.h
    include/ConfigManager.h
    include/TrafficStructures.h
)

# Create main executable
add_executable(trafficflowopt ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(trafficflowopt 
    Threads::Threads
    ${CMAKE_DL_LIBS}
)

# Link optional libraries
if(OpenMP_CXX_FOUND)
    target_link_libraries(trafficflowopt OpenMP::OpenMP_CXX)
    target_compile_definitions(trafficflowopt PRIVATE USE_OPENMP)
endif()

if(Boost_FOUND)
    target_link_libraries(trafficflowopt ${Boost_LIBRARIES})
    target_include_directories(trafficflowopt PRIVATE ${Boost_INCLUDE_DIRS})
    target_compile_definitions(trafficflowopt PRIVATE USE_BOOST)
endif()

# Python integration
find_package(Python3 COMPONENTS Interpreter Development)
if(Python3_FOUND)
    target_link_libraries(trafficflowopt ${Python3_LIBRARIES})
    target_include_directories(trafficflowopt PRIVATE ${Python3_INCLUDE_DIRS})
    target_compile_definitions(trafficflowopt PRIVATE USE_PYTHON)
endif()

# Installation
install(TARGETS trafficflowopt
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES ${HEADERS}
    DESTINATION include/trafficflowopt
)

# Install Python scripts
install(FILES
    python/jax_optimization.py
    python/sensor_api.py
    python/visualization.py
    DESTINATION share/trafficflowopt/python
)

# Install configuration files
install(FILES
    config/traffic_config.json
    config/sensor_config.json
    config/optimization_config.json
    DESTINATION share/trafficflowopt/config
)

# Testing
enable_testing()

# Unit tests
add_subdirectory(tests)

# Performance benchmarks
add_executable(benchmark
    benchmarks/optimization_benchmark.cpp
    ${SOURCES}
)
target_link_libraries(benchmark 
    Threads::Threads
    ${CMAKE_DL_LIBS}
)

# Documentation
find_package(Doxygen)
if(DOXYGEN_FOUND)
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    
    add_custom_target(doc_doxygen ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
endif()

# Package configuration
set(CPACK_PACKAGE_NAME "TrafficFlowOpt")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Intelligent Traffic Optimization System")
set(CPACK_PACKAGE_VENDOR "TrafficFlowOpt Team")
set(CPACK_PACKAGE_CONTACT "support@trafficflowopt.com")

include(CPack)

---

# Dockerfile - Container setup for TrafficFlowOpt
FROM ubuntu:22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    python3 \
    python3-pip \
    python3-dev \
    pkg-config \
    libboost-all-dev \
    libomp-dev \
    curl \
    wget \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Build the application
RUN mkdir -p build && \
    cd build && \
    cmake .. && \
    make -j$(nproc)

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs /app/config

# Copy configuration files
COPY config/ /app/config/

# Set environment variables
ENV JAX_ENABLE_X64=true
ENV CACHE_DURATION=5000
ENV OMP_NUM_THREADS=4

# Expose ports
EXPOSE 8080 8081

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start the application
CMD ["./build/trafficflowopt"]

---

# requirements.txt - Python dependencies
jax[gpu]==0.4.20
jaxlib==0.4.20
numpy==1.24.3
scipy==1.11.1
matplotlib==3.7.2
pandas==2.0.3
plotly==5.15.0
dash==2.13.0
fastapi==0.103.1
uvicorn==0.23.2
pydantic==2.3.0
requests==2.31.0
aiohttp==3.8.5
websockets==11.0.3
pytest==7.4.0
pytest-asyncio==0.21.1
black==23.7.0
flake8==6.0.0
mypy==1.5.1
sphinx==7.1.2
jupyter==1.0.0
notebook==7.0.2
ipywidgets==8.1.0

---

# docker-compose.yml - Multi-service deployment
version: '3.8'

services:
  trafficflowopt:
    build: .
    container_name: trafficflowopt_main
    ports:
      - "8080:8080"
      - "8081:8081"
    environment:
      - JAX_ENABLE_X64=true
      - CACHE_DURATION=5000
      - SENSOR_API_KEY=${SENSOR_API_KEY}
      - GPU_ENABLED=true
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./config:/app/config
    depends_on:
      - redis
      - postgres
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  redis:
    image: redis:7-alpine
    container_name: trafficflowopt_redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    container_name: trafficflowopt_db
    environment:
      - POSTGRES_DB=trafficflowopt
      - POSTGRES_USER=traffic_user
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./sql/init.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    container_name: trafficflowopt_nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
      - ./web:/usr/share/nginx/html
    depends_on:
      - trafficflowopt
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    container_name: trafficflowopt_prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: trafficflowopt_grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
  prometheus_data:
  grafana_data:

networks:
  default:
    name: trafficflowopt_network

---

# .env - Environment configuration template
# Copy to .env and fill in your values

# JAX Configuration
JAX_ENABLE_X64=true
CACHE_DURATION=5000
GPU_ENABLED=true

# API Configuration
SENSOR_API_KEY=your_sensor_api_key_here
SENSOR_API_URL=https://api.trafficflow.com/v1/
API_RATE_LIMIT=60

# Database Configuration
DB_PASSWORD=secure_password_here
DB_HOST=postgres
DB_PORT=5432
DB_NAME=trafficflowopt
DB_USER=traffic_user

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=

# Monitoring
GRAFANA_PASSWORD=admin_password_here
PROMETHEUS_RETENTION=200h

# Application Settings
LOG_LEVEL=INFO
DEBUG_MODE=false
OPTIMIZATION_INTERVAL=600
MAX_CONCURRENT_OPTIMIZATIONS=4

# Security
SECRET_KEY=your_secret_key_here
JWT_EXPIRATION=3600
CORS_ORIGINS=http://localhost:3000,http://localhost:8080

---

# Makefile - Build automation
.PHONY: all build clean test install docker run help

# Default target
all: build

# Build the project
build:
	@echo "Building TrafficFlowOpt..."
	@mkdir -p build
	@cd build && cmake .. && make -j$(nproc)

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf build/
	@find . -name "*.o" -delete
	@find . -name "*.so" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} +

# Run tests
test: build
	@echo "Running tests..."
	@cd build && ctest --output-on-failure

# Install the application
install: build
	@echo "Installing TrafficFlowOpt..."
	@cd build && sudo make install

# Build Docker image
docker:
	@echo "Building Docker image..."
	@docker build -t trafficflowopt:latest .

# Run with Docker Compose
run:
	@echo "Starting TrafficFlowOpt services..."
	@docker-compose up -d

# Stop Docker services
stop:
	@echo "Stopping TrafficFlowOpt services..."
	@docker-compose down

# View logs
logs:
	@docker-compose logs -f trafficflowopt

# Development setup
dev-setup:
	@echo "Setting up development environment..."
	@pip3 install -r requirements.txt
	@pip3 install -r requirements-dev.txt
	@pre-commit install

# Format code
format:
	@echo "Formatting code..."
	@black python/
	@clang-format -i src/*.cpp include/*.h

# Lint code
lint:
	@echo "Linting code..."
	@flake8 python/
	@cppcheck src/ include/

# Generate documentation
docs:
	@echo "Generating documentation..."
	@cd build && make doc_doxygen
	@sphinx-build -b html docs/ docs/_build/

# Benchmark performance
benchmark: build
	@echo "Running performance benchmarks..."
	@cd build && ./benchmark

# Show help
help:
	@echo "TrafficFlowOpt Build System"
	@echo "=========================="
	@echo ""
	@echo "Available targets:"
	@echo "  build       - Build the C++ application"
	@echo "  clean       - Clean build artifacts"
	@echo "  test        - Run unit tests"
	@echo "  install     - Install the application"
	@echo "  docker      - Build Docker image"
	@echo "  run         - Start services with Docker Compose"
	@echo "  stop        - Stop Docker services"
	@echo "  logs        - View application logs"
	@echo "  dev-setup   - Setup development environment"
	@echo "  format      - Format source code"
	@echo "  lint        - Lint source code"
	@echo "  docs        - Generate documentation"
	@echo "  benchmark   - Run performance benchmarks"
	@echo "  help        - Show this help message"

---

# requirements-dev.txt - Development dependencies
pytest==7.4.0
pytest-cov==4.1.0
pytest-asyncio==0.21.1
pytest-mock==3.11.1
black==23.7.0
flake8==6.0.0
mypy==1.5.1
pre-commit==3.3.3
sphinx==7.1.2
sphinx-rtd-theme==1.3.0
coverage==7.3.0
bandit==1.7.5
safety==2.3.4
isort==5.12.0
autopep8==2.0.2

---

# .gitignore - Git ignore patterns
# Build artifacts
build/
dist/
*.o
*.so
*.a
*.lib
*.dll
*.exe

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/
pip-log.txt
pip-delete-this-directory.txt

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Logs
*.log
logs/
*.out

# Configuration
.env
config/local_*.json
*.key
*.pem

# Data
data/
*.csv
*.json.bak
*.db
*.sqlite

# Documentation
docs/_build/
*.pdf

# Testing
.coverage
.pytest_cache/
.tox/
htmlcov/

# Temporary
tmp/
temp/
*.tmp
*.temp
*.cache

# Docker
.dockerignore

---

# install.sh - Installation script
#!/bin/bash

set -e

echo "TrafficFlowOpt Installation Script"
echo "================================="

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "This script should not be run as root" 
   exit 1
fi

# Check system requirements
echo "Checking system requirements..."

# Check for required tools
command -v cmake >/dev/null 2>&1 || { echo >&2 "cmake is required but not installed. Aborting."; exit 1; }
command -v g++ >/dev/null 2>&1 || { echo >&2 "g++ is required but not installed. Aborting."; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo >&2 "python3 is required but not installed. Aborting."; exit 1; }
command -v pip3 >/dev/null 2>&1 || { echo >&2 "pip3 is required but not installed. Aborting."; exit 1; }

echo "✓ System requirements met"

# Install system dependencies (Ubuntu/Debian)
if command -v apt-get >/dev/null 2>&1; then
    echo "Installing system dependencies..."
    sudo apt-get update
    sudo apt-get install -y build-essential cmake g++ python3-dev libboost-all-dev libomp-dev
    echo "✓ System dependencies installed"
fi

# Create directories
echo "Creating directories..."
mkdir -p ~/trafficflowopt/{data,logs,config,backup}
echo "✓ Directories created"

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install --user -r requirements.txt
echo "✓ Python dependencies installed"

# Build the application
echo "Building TrafficFlowOpt..."
mkdir -p build
cd build
cmake ..
make -j$(nproc)
echo "✓ Build completed"

# Run tests
echo "Running tests..."
if make test; then
    echo "✓ All tests passed"
else
    echo "⚠ Some tests failed, but installation will continue"
fi

# Install the application
echo "Installing application..."
sudo make install
echo "✓ Application installed"

# Create systemd service (optional)
read -p "Do you want to install systemd service? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo tee /etc/systemd/system/trafficflowopt.service > /dev/null <<EOF
[Unit]
Description=TrafficFlowOpt - Intelligent Traffic Optimization System
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/trafficflowopt
ExecStart=/usr/local/bin/trafficflowopt
Restart=always
RestartSec=10
Environment=JAX_ENABLE_X64=true
Environment=CACHE_DURATION=5000

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable trafficflowopt
    echo "✓ Systemd service installed"
fi

# Create configuration template
echo "Creating configuration template..."
cat > ~/trafficflowopt/config/traffic_config.json << EOF
{
    "system": {
        "cache_duration": 5000,
        "optimization_interval": 600,
        "max_concurrent_optimizations": 4,
        "log_level": "INFO"
    },
    "optimization": {
        "linear_algebra": {
            "enabled": true,
            "accuracy_threshold": 0.98
        },
        "calculus": {
            "enabled": true,
            "prediction_horizon": 300,
            "time_step": 10
        },
        "graph_theory": {
            "enabled": true,
            "algorithm": "dijkstra",
            "max_alternative_routes": 3
        }
    },
    "jax": {
        "enable_x64": true,
        "platform": "gpu",
        "learning_rate": 0.01,
        "max_iterations": 1000
    },
    "sensors": {
        "api_url": "https://api.trafficflow.com/v1/",
        "api_key": "your_api_key_here",
        "timeout": 30,
        "retry_attempts": 3
    }
}
EOF
echo "✓ Configuration template created"

echo ""
echo "Installation completed successfully!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Edit ~/trafficflowopt/config/traffic_config.json with your settings"
echo "2. Set your sensor API key in the configuration"
echo "3. Run 'trafficflowopt' to start the system"
echo ""
echo "For Docker deployment:"
echo "1. Copy .env.example to .env and configure"
echo "2. Run 'docker-compose up -d'"
echo ""
echo "Documentation: https://github.com/trafficflowopt/docs"
echo "Support: https://github.com/trafficflowopt/issues"