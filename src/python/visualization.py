"""
Traffic Visualization Module
Creates matplotlib/plotly visualizations for traffic analysis
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class TrafficVisualizer:
    """Creates advanced traffic visualizations using matplotlib and plotly"""
    
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def create_traffic_heatmap(self, df: pd.DataFrame, save_path: Optional[str] = None) -> str:
        """Create traffic flow heatmap"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for heatmap
        if 'timestamp' in df.columns and 'hour' in df.columns:
            # Create hour vs day heatmap
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            pivot_data = df.pivot_table(
                values='flow_rate', 
                index='hour', 
                columns='date', 
                aggfunc='mean'
            )
            
            # Take last 7 days for visibility
            if len(pivot_data.columns) > 7:
                pivot_data = pivot_data.iloc[:, -7:]
        else:
            # Create synthetic heatmap data
            hours = range(24)
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            
            # Generate realistic traffic patterns
            data = []
            for hour in hours:
                row = []
                for day_idx in range(len(days)):
                    # Peak hours pattern
                    if 7 <= hour <= 9 or 17 <= hour <= 19:
                        base_flow = 800
                    elif 22 <= hour or hour <= 6:
                        base_flow = 200
                    else:
                        base_flow = 500
                    
                    # Weekend reduction
                    if day_idx >= 5:  # Weekend
                        base_flow *= 0.7
                    
                    # Add variation
                    flow = base_flow * (1 + np.random.normal(0, 0.2))
                    row.append(max(0, flow))
                
                data.append(row)
            
            pivot_data = pd.DataFrame(data, index=hours, columns=days)
        
        # Create heatmap
        sns.heatmap(
            pivot_data, 
            annot=False, 
            cmap='YlOrRd', 
            cbar_kws={'label': 'Average Flow Rate (vehicles/hour)'},
            ax=ax
        )
        
        ax.set_title('Traffic Flow Heatmap - Hourly Patterns', fontsize=16, fontweight='bold')
        ax.set_xlabel('Day', fontsize=12)
        ax.set_ylabel('Hour of Day', fontsize=12)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "traffic_heatmap.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Traffic heatmap saved to {save_path}")
        return str(save_path)
    
    def create_flow_speed_analysis(self, df: pd.DataFrame, save_path: Optional[str] = None) -> str:
        """Create flow vs speed fundamental diagram"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Flow vs Speed scatter plot
        sample_size = min(2000, len(df))
        sample_df = df.sample(sample_size) if len(df) > sample_size else df
        
        axes[0, 0].scatter(
            sample_df['flow_rate'], 
            sample_df['average_speed'],
            alpha=0.6, 
            c=sample_df['occupancy'], 
            cmap='viridis',
            s=20
        )
        axes[0, 0].set_xlabel('Flow Rate (vehicles/hour)')
        axes[0, 0].set_ylabel('Average Speed (km/h)')
        axes[0, 0].set_title('Flow vs Speed Relationship')
        
        # Add theoretical fundamental diagram
        flow_theory = np.linspace(0, df['flow_rate'].max(), 100)
        speed_theory = 80 * (1 - flow_theory / df['flow_rate'].max())  # Simplified model
        axes[0, 0].plot(flow_theory, speed_theory, 'r--', label='Theoretical', linewidth=2)
        axes[0, 0].legend()
        
        # Flow distribution by hour
        if 'hour' in df.columns:
            hourly_flow = df.groupby('hour')['flow_rate'].mean()
            axes[0, 1].bar(hourly_flow.index, hourly_flow.values, color='skyblue', alpha=0.7)
            axes[0, 1].set_xlabel('Hour of Day')
            axes[0, 1].set_ylabel('Average Flow Rate')
            axes[0, 1].set_title('Hourly Traffic Flow Distribution')
            axes[0, 1].set_xticks(range(0, 24, 4))
        
        # Speed distribution
        axes[1, 0].hist(df['average_speed'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 0].axvline(df['average_speed'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df["average_speed"].mean():.1f} km/h')
        axes[1, 0].set_xlabel('Average Speed (km/h)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Speed Distribution')
        axes[1, 0].legend()
        
        # Occupancy vs Flow
        axes[1, 1].scatter(df['occupancy'], df['flow_rate'], alpha=0.6, color='coral')
        axes[1, 1].set_xlabel('Occupancy (%)')
        axes[1, 1].set_ylabel('Flow Rate (vehicles/hour)')
        axes[1, 1].set_title('Occupancy vs Flow Rate')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "flow_speed_analysis.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Flow-speed analysis saved to {save_path}")
        return str(save_path)
    
    def create_network_topology_plot(self, network_data: Dict, save_path: Optional[str] = None) -> str:
        """Create network topology visualization"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Extract nodes and edges
        nodes = network_data.get('nodes', [])
        edges = network_data.get('edges', [])
        
        # Create node positions
        node_positions = {}
        for node in nodes:
            node_positions[node['id']] = (node['lon'], node['lat'])
        
        # Plot edges (roads)
        for edge in edges:
            if edge['from'] in node_positions and edge['to'] in node_positions:
                x1, y1 = node_positions[edge['from']]
                x2, y2 = node_positions[edge['to']]
                
                # Line width proportional to number of lanes
                line_width = edge.get('lanes', 2) * 2
                
                ax.plot([x1, x2], [y1, y2], 'b-', linewidth=line_width, alpha=0.7)
                
                # Add road name
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                ax.annotate(edge['name'], (mid_x, mid_y), fontsize=8, 
                           ha='center', va='center', 
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # Plot nodes (intersections)
        for node in nodes:
            x, y = node['lon'], node['lat']
            
            if node['type'] == 'intersection':
                circle = plt.Circle((x, y), 0.002, color='red', zorder=5)
                ax.add_patch(circle)
                
                # Add node label
                ax.annotate(node['name'], (x, y), xytext=(5, 5), 
                           textcoords='offset points', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Traffic Network Topology', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', lw=4, label='Roads'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=8, label='Intersections')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "network_topology.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Network topology plot saved to {save_path}")
        return str(save_path)
    
    def create_prediction_plot(self, prediction_data: Dict, save_path: Optional[str] = None) -> str:
        """Create traffic prediction visualization"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Extract prediction data
        predictions = prediction_data.get('predictions', {})
        horizon = prediction_data.get('prediction_horizon_minutes', 60)
        
        # Time axis
        time_steps = np.arange(0, horizon, 1)  # Minutes
        
        # Plot flow predictions
        if 'flows' in predictions:
            flows = np.array(predictions['flows'])
            for i in range(min(5, flows.shape[1])):  # Plot up to 5 segments
                axes[0].plot(time_steps, flows[:, i], label=f'Segment {i+1}', linewidth=2)
        
        axes[0].set_xlabel('Time (minutes from now)')
        axes[0].set_ylabel('Flow Rate (vehicles/hour)')
        axes[0].set_title('Predicted Traffic Flow')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot speed predictions
        if 'speeds' in predictions:
            speeds = np.array(predictions['speeds'])
            for i in range(min(5, speeds.shape[1])):
                axes[1].plot(time_steps, speeds[:, i], label=f'Segment {i+1}', linewidth=2)
        
        axes[1].set_xlabel('Time (minutes from now)')
        axes[1].set_ylabel('Average Speed (km/h)')
        axes[1].set_title('Predicted Average Speed')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot occupancy predictions
        if 'occupancies' in predictions:
            occupancies = np.array(predictions['occupancies'])
            for i in range(min(5, occupancies.shape[1])):
                axes[2].plot(time_steps, occupancies[:, i], label=f'Segment {i+1}', linewidth=2)
        
        axes[2].set_xlabel('Time (minutes from now)')
        axes[2].set_ylabel('Occupancy (%)')
        axes[2].set_title('Predicted Occupancy')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Add confidence level
        confidence = prediction_data.get('confidence', 0.8)
        fig.suptitle(f'Traffic Predictions (Confidence: {confidence*100:.1f}%)', 
                     fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "traffic_predictions.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Prediction plot saved to {save_path}")
        return str(save_path)
    
    def create_performance_dashboard(self, performance_data: Dict, save_path: Optional[str] = None) -> str:
        """Create performance metrics dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        timestamps = performance_data.get('timestamps', [])
        metrics = performance_data.get('metrics', {})
        
        # Convert timestamps to datetime
        if timestamps:
            time_axis = [datetime.fromisoformat(ts.replace('Z', '+00:00')) for ts in timestamps]
        else:
            time_axis = [datetime.now() - timedelta(hours=i) for i in range(24, 0, -1)]
        
        # Average Speed over time
        if 'average_speed' in metrics:
            axes[0, 0].plot(time_axis, metrics['average_speed'], 'b-', linewidth=2, marker='o')
            axes[0, 0].set_title('Average Network Speed')
            axes[0, 0].set_ylabel('Speed (km/h)')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Total Flow over time
        if 'total_flow' in metrics:
            axes[0, 1].plot(time_axis, metrics['total_flow'], 'g-', linewidth=2, marker='s')
            axes[0, 1].set_title('Total Network Flow')
            axes[0, 1].set_ylabel('Flow (vehicles/hour)')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Congestion Level over time
        if 'congestion_level' in metrics:
            congestion = np.array(metrics['congestion_level']) * 100  # Convert to percentage
            axes[1, 0].fill_between(time_axis, congestion, alpha=0.7, color='orange')
            axes[1, 0].plot(time_axis, congestion, 'r-', linewidth=2)
            axes[1, 0].set_title('Network Congestion Level')
            axes[1, 0].set_ylabel('Congestion (%)')
            axes[1, 0].set_ylim(0, 100)
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Network Efficiency over time
        if 'efficiency' in metrics:
            efficiency = np.array(metrics['efficiency']) * 100  # Convert to percentage
            axes[1, 1].plot(time_axis, efficiency, 'purple', linewidth=2, marker='^')
            axes[1, 1].set_title('Network Efficiency')
            axes[1, 1].set_ylabel('Efficiency (%)')
            axes[1, 1].set_ylim(0, 100)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Add summary statistics
        summary = performance_data.get('summary', {})
        summary_text = f"""24-Hour Summary:
        Avg Speed: {summary.get('avg_speed_24h', 0):.1f} km/h
        Total Flow: {summary.get('total_flow_24h', 0):.0f} vehicles
        Peak Congestion: {summary.get('peak_congestion', 0)*100:.1f}%
        Avg Efficiency: {summary.get('avg_efficiency', 0)*100:.1f}%"""
        
        fig.text(0.02, 0.98, summary_text, transform=fig.transFigure, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.suptitle('Traffic Network Performance Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "performance_dashboard.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance dashboard saved to {save_path}")
        return str(save_path)
    
    def create_interactive_plotly_dashboard(self, data: Dict) -> str:
        """Create interactive Plotly dashboard for web interface"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Network Flow Heatmap', 'Speed vs Flow', 
                           'Hourly Patterns', 'Prediction Trends'),
            specs=[[{"type": "heatmap"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Sample data for demonstration
        hours = list(range(24))
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        # Generate heatmap data
        heatmap_data = []
        for hour in hours:
            row = []
            for day in range(len(days)):
                if 7 <= hour <= 9 or 17 <= hour <= 19:
                    flow = np.random.uniform(700, 1000)
                else:
                    flow = np.random.uniform(200, 600)
                row.append(flow)
            heatmap_data.append(row)
        
        # Add heatmap
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data,
                x=days,
                y=hours,
                colorscale='Viridis',
                showscale=False
            ),
            row=1, col=1
        )
        
        # Add scatter plot
        flow_vals = np.random.uniform(200, 1000, 100)
        speed_vals = 80 - (flow_vals / 1000) * 60 + np.random.normal(0, 5, 100)
        
        fig.add_trace(
            go.Scatter(
                x=flow_vals,
                y=speed_vals,
                mode='markers',
                name='Data Points',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Add bar chart
        hourly_flows = [np.mean([row[i] for row in heatmap_data]) for i in range(len(days))]
        
        fig.add_trace(
            go.Bar(
                x=days,
                y=hourly_flows,
                name='Daily Average',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Add prediction trend
        time_points = list(range(60))  # Next 60 minutes
        predicted_flow = 500 + 200 * np.sin(np.array(time_points) * 2 * np.pi / 60) + np.random.normal(0, 20, 60)
        
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=predicted_flow,
                mode='lines',
                name='Prediction',
                line=dict(color='red', width=3),
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="TrafficFlowOpt - Interactive Dashboard",
            height=800,
            showlegend=False
        )
        
        # Save as HTML
        output_path = self.output_dir / "interactive_dashboard.html"
        fig.write_html(str(output_path))
        
        logger.info(f"Interactive dashboard saved to {output_path}")
        return str(output_path)