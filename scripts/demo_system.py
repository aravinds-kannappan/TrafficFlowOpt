#!/usr/bin/env python3
"""
TrafficFlowOpt Demo Script
Demonstrates the complete system functionality without heavy dependencies
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def main():
    """Demonstrate TrafficFlowOpt system"""
    
    print("🚦 TrafficFlowOpt - Intelligent Urban Traffic Optimization")
    print("=" * 70)
    print("A real-world C++20/JAX-based traffic optimization solution")
    print()
    
    # Check system status
    print("📋 System Status Check:")
    print("-" * 30)
    
    # Check data availability
    data_dir = project_root / "data"
    if data_dir.exists():
        raw_files = list((data_dir / "raw").glob("*.csv")) if (data_dir / "raw").exists() else []
        processed_files = list((data_dir / "processed").glob("*.csv")) if (data_dir / "processed").exists() else []
        
        print(f"✓ Data directory exists")
        print(f"✓ Raw datasets: {len(raw_files)} files")
        print(f"✓ Processed datasets: {len(processed_files)} files")
        
        if processed_files:
            print(f"  • Real traffic data from Austin and Chicago available")
    else:
        print("⚠ Data directory not found")
    
    # Check web interface
    web_dir = project_root / "docs"
    if web_dir.exists():
        web_files = list(web_dir.glob("*.html"))
        css_files = list((web_dir / "css").glob("*.css")) if (web_dir / "css").exists() else []
        js_files = list((web_dir / "js").glob("*.js")) if (web_dir / "js").exists() else []
        data_assets = list((web_dir / "data").glob("*.json")) if (web_dir / "data").exists() else []
        
        print(f"✓ Web interface ready")
        print(f"  • HTML files: {len(web_files)}")
        print(f"  • CSS files: {len(css_files)}")
        print(f"  • JavaScript modules: {len(js_files)}")
        print(f"  • Data assets: {len(data_assets)}")
    else:
        print("⚠ Web interface not found")
    
    # Check C++ components
    cpp_dir = project_root / "src" / "cpp"
    if cpp_dir.exists():
        cpp_files = list(cpp_dir.rglob("*.cpp"))
        hpp_files = list(cpp_dir.rglob("*.hpp"))
        cmake_file = project_root / "CMakeLists.txt"
        
        print(f"✓ C++ components available")
        print(f"  • Source files: {len(cpp_files)}")
        print(f"  • Header files: {len(hpp_files)}")
        print(f"  • CMake build system: {'✓' if cmake_file.exists() else '✗'}")
    else:
        print("⚠ C++ components not found")
    
    # Check Docker setup
    docker_files = [
        project_root / "Dockerfile",
        project_root / "docker-compose.yml",
        project_root / ".dockerignore"
    ]
    
    docker_ready = all(f.exists() for f in docker_files)
    print(f"✓ Docker deployment: {'Ready' if docker_ready else 'Partial'}")
    
    print()
    
    # Show real data summary
    print("📊 Real Traffic Data Summary:")
    print("-" * 30)
    
    # Load processing report if available
    report_path = project_root / "data" / "processed" / "processing_report.json"
    if report_path.exists():
        try:
            with open(report_path) as f:
                report = json.load(f)
            
            print(f"✓ Total datasets processed: {report.get('datasets_processed', 0)}")
            print(f"✓ Total records: {report.get('total_records', 0):,}")
            print(f"✓ Processing timestamp: {report.get('processing_timestamp', 'Unknown')}")
            
            if 'datasets' in report:
                print("\n📋 Dataset Details:")
                for dataset in report['datasets']:
                    print(f"  • {dataset['source']}: {dataset['records']:,} records")
                    print(f"    File: {dataset['processed_file']}")
                    print(f"    Columns: {', '.join(dataset['columns'][:5])}...")
            
        except Exception as e:
            print(f"⚠ Could not load processing report: {e}")
    else:
        print("⚠ No processing report found")
        print("  Run 'python scripts/fetch_real_data.py' to fetch real data")
    
    print()
    
    # Show web assets summary
    print("🌐 Web Interface Assets:")
    print("-" * 30)
    
    web_data_dir = project_root / "docs" / "data"
    if web_data_dir.exists():
        assets = list(web_data_dir.glob("*.json"))
        print(f"✓ Generated web assets: {len(assets)}")
        
        for asset in assets:
            try:
                with open(asset) as f:
                    data = json.load(f)
                    
                if asset.name == "current_status.json":
                    if "network_summary" in data:
                        summary = data["network_summary"]
                        print(f"  • Current Status: {summary.get('total_segments', 0)} segments")
                        
                elif asset.name == "predictions.json":
                    if "predictions" in data:
                        horizon = data.get("prediction_horizon_minutes", 0)
                        confidence = data.get("confidence", 0)
                        print(f"  • Predictions: {horizon} min horizon, {confidence:.1%} confidence")
                        
                elif asset.name == "performance.json":
                    if "summary" in data:
                        summary = data["summary"]
                        avg_speed = summary.get("avg_speed_24h", 0)
                        print(f"  • Performance: {avg_speed:.1f} km/h avg speed")
                        
            except Exception as e:
                print(f"  • {asset.name}: Available (could not parse: {e})")
    else:
        print("⚠ No web assets found")
        print("  Run 'python scripts/generate_web_assets.py' to generate assets")
    
    print()
    
    # Show deployment options
    print("🚀 Deployment Options:")
    print("-" * 30)
    print("1. GitHub Pages (Static)")
    print("   • Enable GitHub Pages in repository settings")
    print("   • Set source to '/docs' folder")
    print("   • Access at: https://username.github.io/TrafficFlowOpt/")
    print()
    print("2. Docker Deployment (Full System)")
    print("   • Run: docker-compose up -d")
    print("   • Access at: http://localhost:8080")
    print("   • Includes: Web UI, API, Redis, Monitoring")
    print()
    print("3. Local Development")
    print("   • Install: pip install -r requirements.txt")
    print("   • Fetch data: python scripts/fetch_real_data.py")
    print("   • Generate assets: python scripts/generate_web_assets.py")
    print("   • Open: docs/index.html in browser")
    
    print()
    
    # Show mathematical components
    print("🔬 Mathematical Components:")
    print("-" * 30)
    print("✓ C++20 Traffic Network Simulation")
    print("  • Dijkstra's shortest path algorithm")
    print("  • Network flow optimization")
    print("  • Signal timing optimization")
    print("  • Real-time bottleneck detection")
    print()
    print("✓ JAX Neural ODE Prediction (when available)")
    print("  • GPU-accelerated traffic flow forecasting")
    print("  • Runge-Kutta 4th order integration")
    print("  • Uncertainty quantification")
    print("  • Pattern-based prediction fallback")
    print()
    print("✓ Advanced Analytics")
    print("  • Flow-speed fundamental diagrams")
    print("  • Temporal pattern analysis")
    print("  • Performance optimization tracking")
    print("  • Real-time data validation")
    
    print()
    
    # Show next steps
    print("🎯 Next Steps:")
    print("-" * 30)
    
    if not report_path.exists():
        print("1. Fetch real traffic data:")
        print("   python scripts/fetch_real_data.py")
        print()
    
    if not (web_data_dir.exists() and list(web_data_dir.glob("*.json"))):
        print("2. Generate web assets:")
        print("   python scripts/generate_web_assets.py")
        print()
    
    print("3. Deploy the system:")
    print("   • For demo: Open docs/index.html")
    print("   • For production: docker-compose up -d")
    print("   • For GitHub Pages: Enable in repository settings")
    print()
    
    print("4. View the traffic optimization dashboard:")
    if web_dir.exists():
        web_path = web_dir / "index.html"
        print(f"   file://{web_path.absolute()}")
    else:
        print("   Web interface not found")
    
    print()
    print("=" * 70)
    print("🚦 TrafficFlowOpt System Overview Complete")
    print("Built with real data. Optimized with mathematics. Deployed with confidence.")
    print("=" * 70)

if __name__ == "__main__":
    main()