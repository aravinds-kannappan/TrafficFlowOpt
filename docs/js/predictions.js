/**
 * Predictions Manager for TrafficFlowOpt
 * Handles traffic prediction visualization and forecasting
 */

class PredictionsManager {
    constructor() {
        this.charts = {};
        this.currentPredictions = null;
    }

    async updatePredictions(data) {
        try {
            this.currentPredictions = data.predictions;
            this.updateFlowPredictions(data.predictions);
            this.updateSpeedPredictions(data.predictions);
            this.updateCongestionPredictions(data.predictions);
            this.updateConfidenceIndicators(data.predictions);
        } catch (error) {
            console.error('Predictions update failed:', error);
            this.createSamplePredictions();
        }
    }

    updateFlowPredictions(predictionsData) {
        const container = document.getElementById('flow-prediction-chart');
        if (!container) return;

        if (!predictionsData || !predictionsData.predictions) {
            this.createSampleFlowChart();
            return;
        }

        const predictions = predictionsData.predictions;
        const horizon = predictionsData.prediction_horizon_minutes || 60;
        
        // Prepare time axis
        const timeLabels = Array.from({length: Math.min(horizon, 120)}, (_, i) => i);
        const traces = [];

        // Create traces for each segment (up to 5)
        const flowData = predictions.flows || [];
        const segmentCount = Math.min(5, flowData.length > 0 ? flowData[0].length : 0);

        for (let i = 0; i < segmentCount; i++) {
            const segmentFlows = flowData.map(row => row[i] || 0);
            
            traces.push({
                x: timeLabels,
                y: segmentFlows.slice(0, timeLabels.length),
                type: 'scatter',
                mode: 'lines',
                name: `Segment ${i + 1}`,
                line: {
                    width: 2
                }
            });
        }

        const layout = {
            title: {
                text: 'Traffic Flow Predictions',
                font: { size: 16 }
            },
            xaxis: {
                title: 'Time (minutes from now)',
                showgrid: true
            },
            yaxis: {
                title: 'Flow Rate (vehicles/hour)',
                showgrid: true
            },
            margin: { t: 50, l: 60, r: 30, b: 50 },
            height: 350,
            legend: {
                x: 1,
                xanchor: 'right',
                y: 1
            }
        };

        const config = {
            responsive: true,
            displayModeBar: false
        };

        Plotly.newPlot(container, traces, layout, config);
    }

    createSampleFlowChart() {
        const container = document.getElementById('flow-prediction-chart');
        if (!container) return;

        const timeLabels = Array.from({length: 60}, (_, i) => i);
        const traces = [];

        // Generate sample flow predictions for 5 segments
        for (let segment = 0; segment < 5; segment++) {
            const baseFlow = 400 + segment * 100;
            const flowData = timeLabels.map(t => {
                const cyclicPattern = Math.sin(t * Math.PI / 30) * 0.3;
                const trend = -t * 0.5; // Slight downward trend
                const noise = (Math.random() - 0.5) * 50;
                return Math.max(100, baseFlow + baseFlow * cyclicPattern + trend + noise);
            });

            traces.push({
                x: timeLabels,
                y: flowData,
                type: 'scatter',
                mode: 'lines',
                name: `Segment ${segment + 1}`,
                line: { width: 2 }
            });
        }

        const layout = {
            title: {
                text: 'Traffic Flow Predictions (Sample Data)',
                font: { size: 16 }
            },
            xaxis: {
                title: 'Time (minutes from now)',
                showgrid: true
            },
            yaxis: {
                title: 'Flow Rate (vehicles/hour)',
                showgrid: true
            },
            margin: { t: 50, l: 60, r: 30, b: 50 },
            height: 350,
            legend: {
                x: 1,
                xanchor: 'right',
                y: 1
            }
        };

        const config = {
            responsive: true,
            displayModeBar: false
        };

        Plotly.newPlot(container, traces, layout, config);
    }

    updateSpeedPredictions(predictionsData) {
        const container = document.getElementById('speed-prediction-chart');
        if (!container) return;

        if (!predictionsData || !predictionsData.predictions) {
            this.createSampleSpeedChart();
            return;
        }

        const predictions = predictionsData.predictions;
        const horizon = predictionsData.prediction_horizon_minutes || 60;
        const timeLabels = Array.from({length: Math.min(horizon, 120)}, (_, i) => i);
        const traces = [];

        const speedData = predictions.speeds || [];
        const segmentCount = Math.min(5, speedData.length > 0 ? speedData[0].length : 0);

        for (let i = 0; i < segmentCount; i++) {
            const segmentSpeeds = speedData.map(row => row[i] || 0);
            
            traces.push({
                x: timeLabels,
                y: segmentSpeeds.slice(0, timeLabels.length),
                type: 'scatter',
                mode: 'lines',
                name: `Segment ${i + 1}`,
                line: { width: 2 }
            });
        }

        const layout = {
            title: {
                text: 'Speed Predictions',
                font: { size: 16 }
            },
            xaxis: {
                title: 'Time (minutes from now)',
                showgrid: true
            },
            yaxis: {
                title: 'Average Speed (km/h)',
                showgrid: true
            },
            margin: { t: 50, l: 60, r: 30, b: 50 },
            height: 350,
            legend: {
                x: 1,
                xanchor: 'right',
                y: 1
            }
        };

        const config = {
            responsive: true,
            displayModeBar: false
        };

        Plotly.newPlot(container, traces, layout, config);
    }

    createSampleSpeedChart() {
        const container = document.getElementById('speed-prediction-chart');
        if (!container) return;

        const timeLabels = Array.from({length: 60}, (_, i) => i);
        const traces = [];

        for (let segment = 0; segment < 5; segment++) {
            const baseSpeed = 35 + segment * 5;
            const speedData = timeLabels.map(t => {
                const cyclicPattern = Math.cos(t * Math.PI / 45) * 0.2;
                const trend = t * 0.1; // Slight upward trend (traffic improving)
                const noise = (Math.random() - 0.5) * 3;
                return Math.max(15, Math.min(70, baseSpeed + baseSpeed * cyclicPattern + trend + noise));
            });

            traces.push({
                x: timeLabels,
                y: speedData,
                type: 'scatter',
                mode: 'lines',
                name: `Segment ${segment + 1}`,
                line: { width: 2 }
            });
        }

        const layout = {
            title: {
                text: 'Speed Predictions (Sample Data)',
                font: { size: 16 }
            },
            xaxis: {
                title: 'Time (minutes from now)',
                showgrid: true
            },
            yaxis: {
                title: 'Average Speed (km/h)',
                showgrid: true
            },
            margin: { t: 50, l: 60, r: 30, b: 50 },
            height: 350,
            legend: {
                x: 1,
                xanchor: 'right',
                y: 1
            }
        };

        const config = {
            responsive: true,
            displayModeBar: false
        };

        Plotly.newPlot(container, traces, layout, config);
    }

    updateCongestionPredictions(predictionsData) {
        const container = document.getElementById('congestion-prediction-chart');
        if (!container) return;

        if (!predictionsData || !predictionsData.predictions) {
            this.createSampleCongestionChart();
            return;
        }

        const predictions = predictionsData.predictions;
        const horizon = predictionsData.prediction_horizon_minutes || 60;
        const timeLabels = Array.from({length: Math.min(horizon, 120)}, (_, i) => i);

        // Calculate congestion from flow and occupancy data
        const flowData = predictions.flows || [];
        const occupancyData = predictions.occupancies || [];
        
        const traces = [];
        const segmentCount = Math.min(5, Math.max(flowData.length, occupancyData.length));

        for (let i = 0; i < segmentCount; i++) {
            const congestionData = timeLabels.map(t => {
                const flow = flowData[t] ? flowData[t][i] || 0 : 0;
                const occupancy = occupancyData[t] ? occupancyData[t][i] || 0 : 0;
                
                // Simple congestion calculation
                const flowCongestion = Math.min(100, (flow / 800) * 100);
                const occupancyCongestion = Math.min(100, occupancy);
                
                return (flowCongestion + occupancyCongestion) / 2;
            });

            traces.push({
                x: timeLabels,
                y: congestionData,
                type: 'scatter',
                mode: 'lines',
                name: `Segment ${i + 1}`,
                line: { width: 2 },
                fill: i === 0 ? 'tonexty' : undefined,
                fillcolor: i === 0 ? 'rgba(231, 76, 60, 0.1)' : undefined
            });
        }

        const layout = {
            title: {
                text: 'Congestion Level Forecast',
                font: { size: 16 }
            },
            xaxis: {
                title: 'Time (minutes from now)',
                showgrid: true
            },
            yaxis: {
                title: 'Congestion Level (%)',
                showgrid: true,
                range: [0, 100]
            },
            margin: { t: 50, l: 60, r: 30, b: 50 },
            height: 350,
            legend: {
                x: 1,
                xanchor: 'right',
                y: 1
            }
        };

        const config = {
            responsive: true,
            displayModeBar: false
        };

        Plotly.newPlot(container, traces, layout, config);
    }

    createSampleCongestionChart() {
        const container = document.getElementById('congestion-prediction-chart');
        if (!container) return;

        const timeLabels = Array.from({length: 60}, (_, i) => i);
        const traces = [];

        for (let segment = 0; segment < 5; segment++) {
            const baseCongestion = 30 + segment * 10;
            const congestionData = timeLabels.map(t => {
                const cyclicPattern = Math.sin(t * Math.PI / 20 + segment) * 0.4;
                const trend = -t * 0.2; // Gradual improvement
                const noise = (Math.random() - 0.5) * 5;
                return Math.max(0, Math.min(100, baseCongestion + baseCongestion * cyclicPattern + trend + noise));
            });

            traces.push({
                x: timeLabels,
                y: congestionData,
                type: 'scatter',
                mode: 'lines',
                name: `Segment ${segment + 1}`,
                line: { width: 2 },
                fill: segment === 0 ? 'tozeroy' : undefined,
                fillcolor: segment === 0 ? 'rgba(231, 76, 60, 0.1)' : undefined
            });
        }

        const layout = {
            title: {
                text: 'Congestion Level Forecast (Sample Data)',
                font: { size: 16 }
            },
            xaxis: {
                title: 'Time (minutes from now)',
                showgrid: true
            },
            yaxis: {
                title: 'Congestion Level (%)',
                showgrid: true,
                range: [0, 100]
            },
            margin: { t: 50, l: 60, r: 30, b: 50 },
            height: 350,
            legend: {
                x: 1,
                xanchor: 'right',
                y: 1
            }
        };

        const config = {
            responsive: true,
            displayModeBar: false
        };

        Plotly.newPlot(container, traces, layout, config);
    }

    updateConfidenceIndicators(predictionsData) {
        const confidence = predictionsData ? predictionsData.confidence : 0.75;
        
        // Update confidence bars
        const confidenceItems = document.querySelectorAll('.confidence-item');
        
        confidenceItems.forEach((item, index) => {
            const fill = item.querySelector('.confidence-fill');
            const value = item.querySelector('.confidence-value');
            
            if (fill && value) {
                // Decrease confidence over time
                const timeDecay = [1.0, 0.85, 0.65][index] || 0.5;
                const adjustedConfidence = confidence * timeDecay;
                
                fill.style.width = `${adjustedConfidence * 100}%`;
                value.textContent = `${Math.round(adjustedConfidence * 100)}%`;
                
                // Update color based on confidence level
                if (adjustedConfidence > 0.8) {
                    fill.style.background = 'linear-gradient(90deg, #27ae60, #2ecc71)';
                } else if (adjustedConfidence > 0.6) {
                    fill.style.background = 'linear-gradient(90deg, #f39c12, #e67e22)';
                } else {
                    fill.style.background = 'linear-gradient(90deg, #e74c3c, #c0392b)';
                }
            }
        });
    }

    generateNewPredictions() {
        const horizon = parseInt(document.getElementById('prediction-horizon')?.value) || 60;
        
        // Show loading state
        if (window.app) {
            window.app.showLoading();
        }

        // Simulate prediction generation delay
        setTimeout(() => {
            // Generate new sample predictions
            const newPredictions = this.generateSamplePredictions(horizon);
            this.updatePredictions({ predictions: newPredictions });
            
            if (window.app) {
                window.app.hideLoading();
                window.app.showNotification(`New predictions generated for ${horizon} minutes ahead`, 'success');
            }
        }, 1500);
    }

    generateSamplePredictions(horizon) {
        const flows = [];
        const speeds = [];
        const occupancies = [];
        
        for (let t = 0; t < horizon; t++) {
            const flowRow = [];
            const speedRow = [];
            const occupancyRow = [];
            
            for (let s = 0; s < 5; s++) {
                const baseFlow = 400 + s * 100;
                const baseSpeed = 35 + s * 5;
                const baseOccupancy = 40 + s * 8;
                
                const timeVariation = Math.sin(t * Math.PI / 30) * 0.3 + 1;
                const randomFactor = 0.9 + Math.random() * 0.2;
                
                flowRow.push(baseFlow * timeVariation * randomFactor);
                speedRow.push(baseSpeed / timeVariation * randomFactor);
                occupancyRow.push(baseOccupancy * timeVariation * randomFactor);
            }
            
            flows.push(flowRow);
            speeds.push(speedRow);
            occupancies.push(occupancyRow);
        }
        
        return {
            timestamp: new Date().toISOString(),
            prediction_horizon_minutes: horizon,
            predictions: { flows, speeds, occupancies },
            confidence: 0.78 + Math.random() * 0.15
        };
    }

    createSamplePredictions() {
        const sampleData = {
            predictions: this.generateSamplePredictions(60)
        };
        
        this.updatePredictions(sampleData);
    }

    destroy() {
        Object.values(this.charts).forEach(chart => {
            if (chart && chart.destroy) {
                chart.destroy();
            }
        });
        this.charts = {};
    }
}

// Initialize predictions manager
window.predictionsManager = new PredictionsManager();