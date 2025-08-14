# TrafficFlowOpt Deployment Guide 🚀

This guide provides comprehensive instructions for deploying TrafficFlowOpt in various environments.

## 🌐 GitHub Pages Deployment (Recommended for Demo)

GitHub Pages provides free hosting for the web interface with automatic updates.

### Step 1: Repository Setup

1. **Fork the Repository**
   ```bash
   # Fork TrafficFlowOpt to your GitHub account
   # Then clone your fork
   git clone https://github.com/YOUR_USERNAME/TrafficFlowOpt.git
   cd TrafficFlowOpt
   ```

2. **Enable GitHub Pages**
   - Go to your repository on GitHub
   - Navigate to **Settings** → **Pages**
   - Set source to **Deploy from a branch**
   - Select branch: `main` or `master`
   - Select folder: `/docs`
   - Click **Save**

3. **Access Your Deployment**
   - Your site will be available at: `https://YOUR_USERNAME.github.io/TrafficFlowOpt/`
   - Initial deployment may take 5-10 minutes

### Step 2: Data Setup for GitHub Pages

Since GitHub Pages is static hosting, you need to generate data assets:

```bash
# Install dependencies locally
pip install -r requirements.txt

# Fetch real traffic data
python scripts/fetch_real_data.py

# Generate web assets with real data
python scripts/generate_web_assets.py

# Commit and push the generated data
git add docs/data/
git commit -m "Update traffic data assets"
git push origin main
```

### Step 3: Automatic Updates (Optional)

Set up GitHub Actions for automatic data updates:

```yaml
# .github/workflows/update-data.yml
name: Update Traffic Data
on:
  schedule:
    - cron: '0 */6 * * *'  # Every 6 hours
  workflow_dispatch:

jobs:
  update-data:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Fetch real data
      run: python scripts/fetch_real_data.py
    - name: Generate web assets
      run: python scripts/generate_web_assets.py
    - name: Commit updates
      run: |
        git config --local user.email "action@github.com"
        git config --local user.name "GitHub Action"
        git add docs/data/
        git commit -m "Auto-update traffic data" || exit 0
        git push
```

## 🐳 Docker Deployment

Docker provides a complete, self-contained deployment with real-time data fetching.

### Step 1: Prerequisites

- **Docker**: Install [Docker Desktop](https://www.docker.com/products/docker-desktop)
- **Docker Compose**: Usually included with Docker Desktop

### Step 2: Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/TrafficFlowOpt.git
cd TrafficFlowOpt

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f trafficflowopt

# Access the application
open http://localhost:8080
```

### Step 3: Service Overview

The Docker deployment includes:

| Service | Port | Description |
|---------|------|-------------|
| **TrafficFlowOpt** | 8080 | Main web interface |
| **Redis** | 6379 | Data caching |
| **Prometheus** | 9090 | Metrics collection |
| **Grafana** | 3000 | Advanced dashboards |

### Step 4: Production Configuration

For production deployment, create `docker-compose.prod.yml`:

```yaml
version: '3.8'
services:
  trafficflowopt:
    build: .
    ports:
      - "80:80"
    environment:
      - ENVIRONMENT=production
      - DATA_UPDATE_INTERVAL=300
    volumes:
      - traffic_data:/app/data
      - traffic_logs:/app/logs
    restart: always
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G

volumes:
  traffic_data:
  traffic_logs:
```

## ☁️ Cloud Deployment

### AWS Deployment

#### Option 1: AWS ECS (Recommended)

1. **Build and Push Image**
   ```bash
   # Build for AWS
   docker build -t trafficflowopt .
   
   # Tag for ECR
   docker tag trafficflowopt:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/trafficflowopt:latest
   
   # Push to ECR
   docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/trafficflowopt:latest
   ```

2. **ECS Task Definition**
   ```json
   {
     "family": "trafficflowopt",
     "taskRoleArn": "arn:aws:iam::123456789:role/ecsTaskExecutionRole",
     "networkMode": "awsvpc",
     "requiresCompatibilities": ["FARGATE"],
     "cpu": "1024",
     "memory": "2048",
     "containerDefinitions": [
       {
         "name": "trafficflowopt",
         "image": "123456789.dkr.ecr.us-east-1.amazonaws.com/trafficflowopt:latest",
         "portMappings": [
           {
             "containerPort": 80,
             "protocol": "tcp"
           }
         ],
         "environment": [
           {
             "name": "ENVIRONMENT",
             "value": "production"
           }
         ],
         "logConfiguration": {
           "logDriver": "awslogs",
           "options": {
             "awslogs-group": "/ecs/trafficflowopt",
             "awslogs-region": "us-east-1",
             "awslogs-stream-prefix": "ecs"
           }
         }
       }
     ]
   }
   ```

#### Option 2: AWS EC2

```bash
# Launch EC2 instance with Docker
# Install Docker and Docker Compose
sudo yum update -y
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user

# Deploy application
git clone https://github.com/YOUR_USERNAME/TrafficFlowOpt.git
cd TrafficFlowOpt
docker-compose -f docker-compose.prod.yml up -d
```

### Google Cloud Deployment

#### Cloud Run Deployment

```bash
# Build and deploy to Cloud Run
gcloud run deploy trafficflowopt \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --max-instances 10
```

### Azure Deployment

#### Azure Container Instances

```bash
# Create resource group
az group create --name trafficflowopt-rg --location eastus

# Deploy container
az container create \
  --resource-group trafficflowopt-rg \
  --name trafficflowopt \
  --image YOUR_REGISTRY/trafficflowopt:latest \
  --ports 80 \
  --memory 2 \
  --cpu 1 \
  --dns-name-label trafficflowopt
```

## 🔧 Environment Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | development | Deployment environment |
| `DATA_UPDATE_INTERVAL` | 300 | Data refresh interval (seconds) |
| `PREDICTION_HORIZON` | 60 | Prediction time horizon (minutes) |
| `LOG_LEVEL` | INFO | Logging level |
| `REDIS_URL` | redis://localhost:6379 | Redis connection string |
| `DATABASE_URL` | - | Database connection (if used) |

### Configuration Files

#### `config/production.json`
```json
{
  "data_sources": {
    "austin": {
      "url": "https://data.austintexas.gov/resource/...",
      "update_interval": 300,
      "timeout": 30
    },
    "chicago": {
      "url": "https://data.cityofchicago.org/resource/...",
      "update_interval": 600,
      "timeout": 30
    }
  },
  "optimization": {
    "max_iterations": 1000,
    "tolerance": 1e-6,
    "learning_rate": 0.001
  },
  "web_interface": {
    "refresh_interval": 30,
    "map_center": [41.8781, -87.6298],
    "map_zoom": 11
  }
}
```

## 📊 Monitoring and Maintenance

### Health Checks

The application provides several health check endpoints:

- `GET /health` - Basic health status
- `GET /health/ready` - Readiness probe
- `GET /health/live` - Liveness probe
- `GET /metrics` - Prometheus metrics

### Logging

Logs are structured and include:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "component": "data_fetcher",
  "message": "Fetched 1000 records from Austin API",
  "metadata": {
    "source": "austin",
    "records": 1000,
    "duration_ms": 250
  }
}
```

### Metrics

Key metrics exposed:

- `traffic_data_fetch_duration_seconds` - Data fetching time
- `traffic_prediction_accuracy` - Model accuracy
- `traffic_optimization_iterations` - Optimization performance
- `web_requests_total` - HTTP request count
- `system_memory_usage_bytes` - Memory consumption

### Backup and Recovery

#### Data Backup

```bash
# Backup processed data
docker exec trafficflowopt tar -czf /tmp/traffic_backup.tar.gz /app/data/processed

# Copy backup locally
docker cp trafficflowopt:/tmp/traffic_backup.tar.gz ./backup_$(date +%Y%m%d).tar.gz
```

#### Configuration Backup

```bash
# Backup configuration
cp -r config/ backup/config_$(date +%Y%m%d)/
cp docker-compose.yml backup/
```

## 🔒 Security Considerations

### Network Security

- Use HTTPS in production (configure SSL certificates)
- Restrict API access with authentication if needed
- Use VPC/private networks for cloud deployments

### Data Security

- Traffic data is public, but implement rate limiting
- Secure internal communication between services
- Regular security updates for dependencies

### Access Control

```yaml
# nginx security headers (already in docker/nginx.conf)
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header X-Content-Type-Options "nosniff" always;
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Data Fetching Failures

```bash
# Check data fetching logs
docker-compose logs trafficflowopt | grep fetch_real_data

# Manual data fetch
docker exec trafficflowopt python scripts/fetch_real_data.py
```

#### 2. High Memory Usage

```bash
# Check memory usage
docker stats trafficflowopt

# Restart with memory limit
docker-compose down
docker-compose up -d --scale trafficflowopt=1
```

#### 3. Web Interface Not Loading

```bash
# Check nginx logs
docker-compose logs nginx

# Verify file permissions
docker exec trafficflowopt ls -la /app/docs/
```

### Performance Tuning

#### 1. Optimize Data Processing

```python
# In config/production.json
{
  "data_processing": {
    "batch_size": 1000,
    "parallel_workers": 4,
    "cache_ttl": 300
  }
}
```

#### 2. Database Optimization

```bash
# Add database indices for common queries
# Configure connection pooling
# Implement data partitioning for large datasets
```

## 📞 Support

### Getting Help

1. **Documentation**: Check this deployment guide and main README
2. **Issues**: Create GitHub issues for bugs or feature requests
3. **Discussions**: Use GitHub Discussions for questions
4. **Logs**: Always include relevant logs when reporting issues

### Performance Monitoring

Set up alerts for:

- High memory usage (>1.5GB)
- Data fetch failures
- API response time >500ms
- Prediction accuracy drops <80%

---

**Deployment complete! Your TrafficFlowOpt system is now running with real traffic data.** 🎉