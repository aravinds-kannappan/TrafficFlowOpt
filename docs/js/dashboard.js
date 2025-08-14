/**
 * Dashboard Manager for TrafficFlowOpt
 * Handles real-time dashboard updates and visualizations
 */

class DashboardManager {
    constructor() {
        this.charts = {};
        this.metricsUpdateInterval = null;
    }

    async updateDashboard(data) {
        try {
            this.updateMetrics(data.currentStatus);
            this.updateHeatmap(data.performance);
            this.updatePerformanceChart(data.performance);
            this.updateNetworkStatus(data.currentStatus);
        } catch (error) {
            console.error('Dashboard update failed:', error);
        }
    }

    updateMetrics(currentStatus) {
        if (!currentStatus || !currentStatus.network_summary) {
            this.setMetricsToDefault();
            return;
        }

        const summary = currentStatus.network_summary;
        
        // Update metric values with animation
        this.animateMetricValue('total-flow', summary.average_flow * summary.total_segments, 0);
        this.animateMetricValue('avg-speed', summary.average_speed, 1);
        this.animateMetricValue('congestion-level', (summary.congested_segments / summary.total_segments) * 100, 1);
        this.animateMetricValue('efficiency', this.calculateEfficiency(summary), 1);
    }

    setMetricsToDefault() {
        document.getElementById('total-flow').textContent = '--';
        document.getElementById('avg-speed').textContent = '--';
        document.getElementById('congestion-level').textContent = '--';
        document.getElementById('efficiency').textContent = '--';
    }

    animateMetricValue(elementId, targetValue, decimals) {
        const element = document.getElementById(elementId);
        if (!element) return;

        const currentValue = parseFloat(element.textContent.replace(/[^0-9.-]/g, '')) || 0;
        const duration = 1000; // 1 second animation
        const startTime = performance.now();

        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Easing function for smooth animation
            const easeOutCubic = 1 - Math.pow(1 - progress, 3);
            const animatedValue = currentValue + (targetValue - currentValue) * easeOutCubic;
            
            element.textContent = animatedValue.toFixed(decimals);
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            } else {
                element.textContent = targetValue.toFixed(decimals);
            }
        };

        requestAnimationFrame(animate);
    }

    calculateEfficiency(summary) {
        // Simple efficiency calculation based on speed and congestion
        if (!summary.average_speed || !summary.total_segments) return 0;
        
        const speedEfficiency = Math.min(summary.average_speed / 60, 1); // Normalize to max 60 km/h
        const congestionPenalty = summary.congested_segments / summary.total_segments;
        
        return (speedEfficiency * (1 - congestionPenalty)) * 100;
    }

    updateHeatmap(performanceData) {
        if (!performanceData || !performanceData.metrics) {
            this.createSampleHeatmap();
            return;
        }

        const container = document.getElementById('heatmap-chart');
        if (!container) return;

        // Generate heatmap data from performance metrics
        const heatmapData = this.generateHeatmapData(performanceData);
        
        const data = [{
            z: heatmapData.values,
            x: heatmapData.hours,
            y: heatmapData.days,
            type: 'heatmap',
            colorscale: [
                [0, '#27ae60'],      // Green for low traffic
                [0.5, '#f39c12'],    // Orange for medium traffic
                [1, '#e74c3c']       // Red for high traffic
            ],
            showscale: true,
            colorbar: {
                title: 'Traffic Flow<br>(vehicles/hour)',
                titleside: 'right'
            }
        }];

        const layout = {
            title: {
                text: 'Traffic Flow by Hour and Day',
                font: { size: 16 }
            },
            xaxis: {
                title: 'Hour of Day',
                dtick: 2
            },
            yaxis: {
                title: 'Day of Week'
            },
            margin: { t: 50, l: 80, r: 80, b: 50 },
            height: 350
        };

        const config = {
            responsive: true,
            displayModeBar: false
        };

        Plotly.newPlot(container, data, layout, config);
    }

    createSampleHeatmap() {
        const container = document.getElementById('heatmap-chart');
        if (!container) return;

        // Generate sample data
        const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
        const hours = Array.from({length: 24}, (_, i) => i);
        const values = [];

        for (let day = 0; day < 7; day++) {
            const dayValues = [];
            for (let hour = 0; hour < 24; hour++) {
                let flow;
                if ((hour >= 7 && hour <= 9) || (hour >= 17 && hour <= 19)) {
                    // Peak hours
                    flow = 800 + Math.random() * 200;
                } else if (hour >= 22 || hour <= 6) {
                    // Night hours
                    flow = 100 + Math.random() * 100;
                } else {
                    // Regular hours
                    flow = 400 + Math.random() * 200;
                }
                
                // Weekend reduction
                if (day === 0 || day === 6) {
                    flow *= 0.7;
                }
                
                dayValues.push(Math.round(flow));
            }
            values.push(dayValues);
        }

        const data = [{
            z: values,
            x: hours,
            y: days,
            type: 'heatmap',
            colorscale: [
                [0, '#27ae60'],
                [0.5, '#f39c12'],
                [1, '#e74c3c']
            ],
            showscale: true,
            colorbar: {
                title: 'Traffic Flow<br>(vehicles/hour)',
                titleside: 'right'
            }
        }];

        const layout = {
            title: {
                text: 'Traffic Flow Pattern (Sample Data)',
                font: { size: 16 }
            },
            xaxis: {
                title: 'Hour of Day',
                dtick: 2
            },
            yaxis: {
                title: 'Day of Week'
            },
            margin: { t: 50, l: 80, r: 80, b: 50 },
            height: 350
        };

        const config = {
            responsive: true,
            displayModeBar: false
        };

        Plotly.newPlot(container, data, layout, config);
    }

    generateHeatmapData(performanceData) {
        const hours = Array.from({length: 24}, (_, i) => i);
        const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
        
        // Use recent performance data to estimate patterns
        const flowData = performanceData.metrics.total_flow || [];
        const timestamps = performanceData.timestamps || [];
        
        const values = [];
        
        for (let day = 0; day < 7; day++) {
            const dayValues = [];
            for (let hour = 0; hour < 24; hour++) {
                // Find matching data points or use pattern-based estimation
                let flow = this.estimateFlowForTimeSlot(day, hour, flowData, timestamps);
                dayValues.push(Math.round(flow));
            }
            values.push(dayValues);
        }
        
        return { values, hours, days };
    }

    estimateFlowForTimeSlot(day, hour, flowData, timestamps) {
        // Try to find actual data for this time slot
        if (timestamps.length > 0 && flowData.length > 0) {
            const recentData = timestamps.map((ts, idx) => ({
                timestamp: new Date(ts),
                flow: flowData[idx]
            }));
            
            // Find data points for this hour
            const hourMatches = recentData.filter(d => d.timestamp.getHours() === hour);
            if (hourMatches.length > 0) {
                return hourMatches.reduce((sum, d) => sum + d.flow, 0) / hourMatches.length;
            }
        }
        
        // Fallback to pattern-based estimation
        let baseFlow = 400;
        
        // Hour patterns
        if ((hour >= 7 && hour <= 9) || (hour >= 17 && hour <= 19)) {
            baseFlow = 800; // Peak hours
        } else if (hour >= 22 || hour <= 6) {
            baseFlow = 200; // Night hours
        }
        
        // Day patterns
        if (day === 0 || day === 6) {
            baseFlow *= 0.7; // Weekend reduction
        }
        
        return baseFlow + Math.random() * 100 - 50;
    }

    updatePerformanceChart(performanceData) {
        if (!performanceData || !performanceData.metrics) {
            this.createSamplePerformanceChart();
            return;
        }

        const canvas = document.getElementById('performance-chart');
        if (!canvas) return;

        // Destroy existing chart
        if (this.charts.performance) {
            this.charts.performance.destroy();
        }

        const ctx = canvas.getContext('2d');
        const timestamps = performanceData.timestamps || [];
        const metrics = performanceData.metrics;

        // Prepare labels (last 12 hours)
        const labels = timestamps.slice(-12).map(ts => {
            const date = new Date(ts);
            return `${date.getHours().toString().padStart(2, '0')}:00`;
        });

        const data = {
            labels: labels,
            datasets: [
                {
                    label: 'Average Speed (km/h)',
                    data: (metrics.average_speed || []).slice(-12),
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    tension: 0.4,
                    yAxisID: 'y'
                },
                {
                    label: 'Congestion Level (%)',
                    data: (metrics.congestion_level || []).slice(-12).map(c => c * 100),
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    tension: 0.4,
                    yAxisID: 'y1'
                }
            ]
        };

        const config = {
            type: 'line',
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Performance Trends (Last 12 Hours)'
                    },
                    legend: {
                        position: 'top'
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Speed (km/h)'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Congestion (%)'
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                    }
                }
            }
        };

        this.charts.performance = new Chart(ctx, config);
    }

    createSamplePerformanceChart() {
        const canvas = document.getElementById('performance-chart');
        if (!canvas) return;

        if (this.charts.performance) {
            this.charts.performance.destroy();
        }

        const ctx = canvas.getContext('2d');
        
        // Generate sample data for last 12 hours
        const now = new Date();
        const labels = [];
        const speedData = [];
        const congestionData = [];

        for (let i = 11; i >= 0; i--) {
            const time = new Date(now.getTime() - i * 60 * 60 * 1000);
            labels.push(`${time.getHours().toString().padStart(2, '0')}:00`);
            
            const hour = time.getHours();
            let speed, congestion;
            
            if ((hour >= 7 && hour <= 9) || (hour >= 17 && hour <= 19)) {
                speed = 25 + Math.random() * 15; // Peak hours - slower
                congestion = 60 + Math.random() * 30;
            } else {
                speed = 45 + Math.random() * 20; // Off-peak - faster
                congestion = 20 + Math.random() * 30;
            }
            
            speedData.push(speed);
            congestionData.push(congestion);
        }

        const data = {
            labels: labels,
            datasets: [
                {
                    label: 'Average Speed (km/h)',
                    data: speedData,
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    tension: 0.4,
                    yAxisID: 'y'
                },
                {
                    label: 'Congestion Level (%)',
                    data: congestionData,
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    tension: 0.4,
                    yAxisID: 'y1'
                }
            ]
        };

        const config = {
            type: 'line',
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Performance Trends (Sample Data)'
                    },
                    legend: {
                        position: 'top'
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Speed (km/h)'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Congestion (%)'
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                    }
                }
            }
        };

        this.charts.performance = new Chart(ctx, config);
    }

    updateNetworkStatus(currentStatus) {
        const container = document.getElementById('network-status-grid');
        if (!container) return;

        if (!currentStatus || !currentStatus.segments) {
            this.createSampleNetworkStatus(container);
            return;
        }

        container.innerHTML = '';

        currentStatus.segments.forEach(segment => {
            const statusItem = document.createElement('div');
            statusItem.className = `status-item ${segment.status}`;
            
            statusItem.innerHTML = `
                <div class="status-name">${segment.name}</div>
                <div class="status-metrics">
                    <div class="status-value">Flow: ${Math.round(segment.flow_rate)} veh/h</div>
                    <div class="status-value">Speed: ${segment.average_speed.toFixed(1)} km/h</div>
                    <div class="status-value">Occupancy: ${Math.round(segment.occupancy)}%</div>
                </div>
            `;
            
            container.appendChild(statusItem);
        });
    }

    createSampleNetworkStatus(container) {
        const sampleSegments = [
            { name: 'Broadway 42nd', flow_rate: 720, average_speed: 38.5, occupancy: 68, status: 'warning' },
            { name: '7th Ave 42nd', flow_rate: 580, average_speed: 52.1, occupancy: 55, status: 'normal' },
            { name: 'Times Square', flow_rate: 950, average_speed: 25.3, occupancy: 85, status: 'error' },
            { name: 'Herald Square', flow_rate: 420, average_speed: 45.8, occupancy: 42, status: 'normal' }
        ];

        container.innerHTML = '';

        sampleSegments.forEach(segment => {
            const statusItem = document.createElement('div');
            statusItem.className = `status-item ${segment.status}`;
            
            statusItem.innerHTML = `
                <div class="status-name">${segment.name}</div>
                <div class="status-metrics">
                    <div class="status-value">Flow: ${Math.round(segment.flow_rate)} veh/h</div>
                    <div class="status-value">Speed: ${segment.average_speed.toFixed(1)} km/h</div>
                    <div class="status-value">Occupancy: ${Math.round(segment.occupancy)}%</div>
                </div>
            `;
            
            container.appendChild(statusItem);
        });
    }

    destroy() {
        if (this.metricsUpdateInterval) {
            clearInterval(this.metricsUpdateInterval);
        }
        
        Object.values(this.charts).forEach(chart => {
            if (chart && chart.destroy) {
                chart.destroy();
            }
        });
        
        this.charts = {};
    }
}

// Initialize dashboard manager
window.dashboardManager = new DashboardManager();