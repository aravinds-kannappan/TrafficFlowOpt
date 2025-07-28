# api_server.py - FastAPI REST API server for TrafficFlowOpt
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
import uvicorn

# Import our modules
from jax_optimization import JAXTrafficOptimizer, TrafficNode, OptimizationConfig
import subprocess
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# FastAPI app
app = FastAPI(
    title="TrafficFlowOpt API",
    description="Intelligent Traffic Optimization System REST API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global optimizer instance
optimizer: Optional[JAXTrafficOptimizer] = None
optimization_status = {
    "is_running": False,
    "last_run": None,
    "next_run": None,
    "results": None
}

# Pydantic models
class TrafficNodeModel(BaseModel):
    node_id: int = Field(..., ge=0)
    density: float = Field(..., ge=0.0, le=1.0)
    flow_rate: float = Field(..., ge=0.0)
    connected_nodes: List[int] = Field(default_factory=list)
    signal_timing: float = Field(default=45.0, ge=20.0, le=120.0)

class OptimizationRequest(BaseModel):
    traffic_nodes: List[TrafficNodeModel]
    config: Optional[Dict[str, Any]] = None
    prediction_horizon: Optional[int] = Field(default=300, ge=60, le=3600)

class OptimizationResponse(BaseModel):
    optimization_id: str
    status: str
    results: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    timestamp: datetime

class PredictionRequest(BaseModel):
    traffic_nodes: List[TrafficNodeModel]
    prediction_horizon: int = Field(default=300, ge=60, le=3600)

class SystemStatus(BaseModel):
    status: str
    uptime: float
    optimization_status: Dict[str, Any]
    system_metrics: Dict[str, Any]

class ConfigUpdate(BaseModel):
    learning_rate: Optional[float] = Field(None, gt=0.0, le=1.0)
    max_iterations: Optional[int] = Field(None, ge=100, le=10000)
    convergence_threshold: Optional[float] = Field(None, gt=0.0)
    regularization: Optional[float] = Field(None, ge=0.0)

# Startup event
@app.on_event("startup")
async def startup_event():
    global optimizer
    logger.info("Starting TrafficFlowOpt API server...")
    
    # Initialize JAX optimizer
    config = OptimizationConfig()
    optimizer = JAXTrafficOptimizer(config)
    
    # Start background optimization scheduler
    asyncio.create_task(optimization_scheduler())
    
    logger.info("TrafficFlowOpt API server started successfully")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down TrafficFlowOpt API server...")

# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # In production, implement proper JWT token validation
    token = credentials.credentials
    if token != os.getenv("API_TOKEN", "demo_token"):
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return {"user_id": "api_user", "token": token}

# Background optimization scheduler
async def optimization_scheduler():
    """Background task to run periodic optimizations"""
    while True:
        try:
            interval = int(os.getenv("OPTIMIZATION_INTERVAL", "600"))  # 10 minutes default
            await asyncio.sleep(interval)
            
            if not optimization_status["is_running"]:
                logger.info("Running scheduled optimization...")
                # Run optimization with default traffic data
                sample_nodes = create_sample_traffic_nodes()
                await run_optimization_background(sample_nodes)
                
        except Exception as e:
            logger.error(f"Error in optimization scheduler: {e}")

def create_sample_traffic_nodes() -> List[TrafficNode]:
    """Create sample traffic data for scheduled optimizations"""
    import numpy as np
    nodes = []
    for i in range(64):
        row, col = i // 8, i % 8
        distance_from_center = np.sqrt((row - 3.5)**2 + (col - 3.5)**2)
        base_density = 0.3 + 0.4 * np.exp(-distance_from_center / 3.0)
        density = np.clip(base_density + np.random.normal(0, 0.1), 0.1, 0.9)
        flow_rate = 100 * (1.0 - density) + np.random.normal(0, 10)
        flow_rate = np.clip(flow_rate, 20, 120)
        
        connected = []
        if row > 0: connected.append((row-1) * 8 + col)
        if row < 7: connected.append((row+1) * 8 + col)
        if col > 0: connected.append(row * 8 + (col-1))
        if col < 7: connected.append(row * 8 + (col+1))
        
        nodes.append(TrafficNode(
            node_id=i, density=density, flow_rate=flow_rate,
            connected_nodes=connected, signal_timing=45.0
        ))
    return nodes

async def run_optimization_background(traffic_nodes: List[TrafficNode]):
    """Run optimization in background"""
    global optimization_status
    
    optimization_status["is_running"] = True
    optimization_status["last_run"] = datetime.now()
    
    try:
        start_time = time.time()
        results = optimizer.optimize_traffic_flow(traffic_nodes)
        execution_time = time.time() - start_time
        
        optimization_status["results"] = results
        optimization_status["execution_time"] = execution_time
        logger.info(f"Background optimization completed in {execution_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Background optimization failed: {e}")
    finally:
        optimization_status["is_running"] = False
        optimization_status["next_run"] = datetime.now() + timedelta(
            seconds=int(os.getenv("OPTIMIZATION_INTERVAL", "600"))
        )

# API Routes

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "TrafficFlowOpt API",
        "version": "1.0.0",
        "description": "Intelligent Traffic Optimization System",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "optimizer_ready": optimizer is not None,
        "uptime": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
    }

@app.get("/status", response_model=SystemStatus)
async def get_system_status(user: dict = Depends(get_current_user)):
    """Get comprehensive system status"""
    uptime = time.time() - getattr(app.state, 'start_time', time.time())
    
    # Get system metrics
    try:
        import psutil
        system_metrics = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
    except ImportError:
        system_metrics = {"note": "psutil not available"}
    
    return SystemStatus(
        status="operational",
        uptime=uptime,
        optimization_status=optimization_status,
        system_metrics=system_metrics
    )

@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_traffic(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user)
):
    """Run traffic flow optimization"""
    if not optimizer:
        raise HTTPException(status_code=500, detail="Optimizer not initialized")
    
    optimization_id = f"opt_{int(time.time())}"
    logger.info(f"Starting optimization {optimization_id}")
    
    try:
        # Convert Pydantic models to TrafficNode objects
        traffic_nodes = [
            TrafficNode(
                node_id=node.node_id,
                density=node.density,
                flow_rate=node.flow_rate,
                connected_nodes=node.connected_nodes,
                signal_timing=node.signal_timing
            )
            for node in request.traffic_nodes
        ]
        
        # Update optimizer config if provided
        if request.config:
            current_config = optimizer.config
            for key, value in request.config.items():
                if hasattr(current_config, key):
                    setattr(current_config, key, value)
        
        # Run optimization
        start_time = time.time()
        results = optimizer.optimize_traffic_flow(traffic_nodes)
        execution_time = time.time() - start_time
        
        logger.info(f"Optimization {optimization_id} completed in {execution_time:.2f}s")
        
        return OptimizationResponse(
            optimization_id=optimization_id,
            status="completed",
            results=results,
            execution_time=execution_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Optimization {optimization_id} failed: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@app.post("/predict", response_model=Dict[str, Any])
async def predict_traffic(
    request: PredictionRequest,
    user: dict = Depends(get_current_user)
):
    """Predict future traffic flow patterns"""
    if not optimizer:
        raise HTTPException(status_code=500, detail="Optimizer not initialized")
    
    try:
        # Convert to TrafficNode objects
        traffic_nodes = [
            TrafficNode(
                node_id=node.node_id,
                density=node.density,
                flow_rate=node.flow_rate,
                connected_nodes=node.connected_nodes,
                signal_timing=node.signal_timing
            )
            for node in request.traffic_nodes
        ]
        
        # Run prediction
        start_time = time.time()
        results = optimizer.predict_traffic_flow(traffic_nodes, request.prediction_horizon)
        execution_time = time.time() - start_time
        
        return {
            "prediction_id": f"pred_{int(time.time())}",
            "status": "completed",
            "results": results,
            "execution_time": execution_time,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/optimization/history", response_model=List[Dict[str, Any]])
async def get_optimization_history(
    limit: int = 10,
    user: dict = Depends(get_current_user)
):
    """Get optimization history"""
    if not optimizer:
        raise HTTPException(status_code=500, detail="Optimizer not initialized")
    
    history = optimizer.optimization_history[-limit:] if optimizer.optimization_history else []
    return history

@app.put("/config", response_model=Dict[str, str])
async def update_config(
    config_update: ConfigUpdate,
    user: dict = Depends(get_current_user)
):
    """Update optimization configuration"""
    if not optimizer:
        raise HTTPException(status_code=500, detail="Optimizer not initialized")
    
    try:
        current_config = optimizer.config
        updated_fields = []
        
        for field, value in config_update.dict(exclude_none=True).items():
            if hasattr(current_config, field):
                setattr(current_config, field, value)
                updated_fields.append(field)
        
        return {
            "status": "success",
            "message": f"Updated configuration fields: {', '.join(updated_fields)}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Config update failed: {str(e)}")

@app.get("/config", response_model=Dict[str, Any])
async def get_config(user: dict = Depends(get_current_user)):
    """Get current optimization configuration"""
    if not optimizer:
        raise HTTPException(status_code=500, detail="Optimizer not initialized")
    
    config = optimizer.config
    return {
        "learning_rate": config.learning_rate,
        "max_iterations": config.max_iterations,
        "convergence_threshold": config.convergence_threshold,
        "regularization": config.regularization,
        "batch_size": config.batch_size
    }

@app.post("/cpp/optimize", response_model=Dict[str, Any])
async def run_cpp_optimization(
    traffic_data: List[Dict[str, Any]],
    user: dict = Depends(get_current_user)
):
    """Run C++ backend optimization"""
    try:
        # Write traffic data to temporary file
        temp_file = f"/tmp/traffic_data_{int(time.time())}.json"
        with open(temp_file, 'w') as f:
            json.dump(traffic_data, f)
        
        # Run C++ optimizer
        cmd = ["/usr/local/bin/trafficflowopt", "--input", temp_file, "--output", f"{temp_file}.out"]
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if process.returncode != 0:
            raise Exception(f"C++ optimization failed: {process.stderr}")
        
        # Read results
        with open(f"{temp_file}.out", 'r') as f:
            results = json.load(f)
        
        # Cleanup
        os.remove(temp_file)
        os.remove(f"{temp_file}.out")
        
        return {
            "status": "completed",
            "backend": "cpp",
            "results": results,
            "timestamp": datetime.now()
        }
        
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="C++ optimization timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"C++ optimization failed: {str(e)}")

@app.get("/metrics", response_model=Dict[str, Any])
async def get_metrics():
    """Get system metrics for monitoring"""
    try:
        import psutil
        
        metrics = {
            "timestamp": datetime.now(),
            "system": {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory": {
                    "percent": psutil.virtual_memory().percent,
                    "available": psutil.virtual_memory().available,
                    "used": psutil.virtual_memory().used
                },
                "disk": {
                    "percent": psutil.disk_usage('/').percent,
                    "free": psutil.disk_usage('/').free,
                    "used": psutil.disk_usage('/').used
                }
            },
            "optimization": optimization_status,
            "api": {
                "uptime": time.time() - getattr(app.state, 'start_time', time.time()),
                "optimizer_ready": optimizer is not None
            }
        }
        
        return metrics
        
    except ImportError:
        return {
            "timestamp": datetime.now(),
            "system": {"note": "System metrics unavailable (psutil not installed)"},
            "optimization": optimization_status,
            "api": {
                "uptime": time.time() - getattr(app.state, 'start_time', time.time()),
                "optimizer_ready": optimizer is not None
            }
        }

# WebSocket endpoint for real-time updates
@app.websocket("/ws/optimization")
async def websocket_optimization(websocket):
    """WebSocket endpoint for real-time optimization updates"""
    await websocket.accept()
    
    try:
        while True:
            # Send current optimization status
            status_update = {
                "type": "status_update",
                "data": optimization_status,
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send_json(status_update)
            
            # Wait for 5 seconds before next update
            await asyncio.sleep(5)
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    # Set start time
    import time
    app.state.start_time = time.time()
    
    # Run the server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8080,
        reload=os.getenv("DEBUG_MODE", "false").lower() == "true",
        workers=1,
        log_level="info"
    )

---

# config_manager.py - Configuration management system
import json
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import yaml

logger = logging.getLogger(__name__)

@dataclass
class SystemConfig:
    """System-wide configuration"""
    cache_duration: int = 5000
    optimization_interval: int = 600
    max_concurrent_optimizations: int = 4
    log_level: str = "INFO"
    debug_mode: bool = False

@dataclass
class JAXConfig:
    """JAX optimization configuration"""
    enable_x64: