/**
 * TrafficFlowOpt - Main Application JavaScript
 * Handles navigation, data loading, and overall application state
 */

class TrafficFlowOptApp {
    constructor() {
        this.currentSection = 'dashboard';
        this.data = {
            currentStatus: null,
            predictions: null,
            network: null,
            performance: null
        };
        this.updateInterval = null;
        this.init();
    }

    async init() {
        this.setupNavigation();
        this.setupEventListeners();
        await this.loadInitialData();
        this.startAutoRefresh();
        this.hideLoading();
    }

    setupNavigation() {
        const navLinks = document.querySelectorAll('.nav-link');
        const sections = document.querySelectorAll('.section');

        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const targetSection = link.getAttribute('href').substring(1);
                this.showSection(targetSection);
                
                // Update active nav link
                navLinks.forEach(l => l.classList.remove('active'));
                link.classList.add('active');
            });
        });
    }

    showSection(sectionId) {
        const sections = document.querySelectorAll('.section');
        sections.forEach(section => section.classList.remove('active'));
        
        const targetSection = document.getElementById(sectionId);
        if (targetSection) {
            targetSection.classList.add('active');
            this.currentSection = sectionId;
            
            // Load section-specific data
            this.loadSectionData(sectionId);
        }
    }

    async loadSectionData(sectionId) {
        switch (sectionId) {
            case 'dashboard':
                await this.loadDashboardData();
                break;
            case 'network':
                await this.loadNetworkData();
                break;
            case 'predictions':
                await this.loadPredictionData();
                break;
            case 'analytics':
                await this.loadAnalyticsData();
                break;
        }
    }

    setupEventListeners() {
        // Global error handling
        window.addEventListener('error', (e) => {
            console.error('Application error:', e.error);
            this.showNotification('An error occurred. Please refresh the page.', 'error');
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch (e.key) {
                    case '1':
                        e.preventDefault();
                        this.showSection('dashboard');
                        break;
                    case '2':
                        e.preventDefault();
                        this.showSection('network');
                        break;
                    case '3':
                        e.preventDefault();
                        this.showSection('predictions');
                        break;
                    case '4':
                        e.preventDefault();
                        this.showSection('analytics');
                        break;
                    case 'r':
                        e.preventDefault();
                        this.refreshCurrentSection();
                        break;
                }
            }
        });
    }

    async loadInitialData() {
        try {
            this.showLoading();
            
            // Load all data in parallel
            const [currentStatus, predictions, network, performance] = await Promise.all([
                this.fetchData('data/current_status.json'),
                this.fetchData('data/predictions.json'),
                this.fetchData('data/network.json'),
                this.fetchData('data/performance.json')
            ]);

            this.data.currentStatus = currentStatus;
            this.data.predictions = predictions;
            this.data.network = network;
            this.data.performance = performance;

            // Update last updated time
            this.updateLastUpdatedTime();
            
            // Load initial dashboard
            await this.loadDashboardData();

        } catch (error) {
            console.error('Failed to load initial data:', error);
            this.showNotification('Failed to load traffic data. Using fallback data.', 'warning');
            this.loadFallbackData();
        }
    }

    async fetchData(url) {
        try {
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return await response.json();
        } catch (error) {
            console.warn(`Failed to fetch ${url}:`, error);
            return this.generateFallbackData(url);
        }
    }

    generateFallbackData(url) {
        const filename = url.split('/').pop().split('.')[0];
        
        switch (filename) {
            case 'current_status':
                return this.generateMockCurrentStatus();
            case 'predictions':
                return this.generateMockPredictions();
            case 'network':
                return this.generateMockNetwork();
            case 'performance':
                return this.generateMockPerformance();
            default:
                return {};
        }
    }

    generateMockCurrentStatus() {
        return {
            timestamp: new Date().toISOString(),
            network_summary: {
                total_segments: 5,
                average_flow: 650,
                average_speed: 45.2,
                average_occupancy: 62,
                congested_segments: 2
            },
            segments: [
                {
                    id: "Broadway_42nd",
                    name: "Broadway 42nd",
                    flow_rate: 720,
                    average_speed: 38.5,
                    occupancy: 68,
                    status: "congested",
                    latitude: 40.7589,
                    longitude: -73.9851
                },
                {
                    id: "7th_Ave_42nd",
                    name: "7th Ave 42nd",
                    flow_rate: 580,
                    average_speed: 52.1,
                    occupancy: 55,
                    status: "normal",
                    latitude: 40.7505,
                    longitude: -73.9934
                }
            ]
        };
    }

    generateMockPredictions() {
        const timeSteps = 60;
        const segments = 5;
        
        const flows = [];
        const speeds = [];
        const occupancies = [];
        
        for (let t = 0; t < timeSteps; t++) {
            const flowRow = [];
            const speedRow = [];
            const occupancyRow = [];
            
            for (let s = 0; s < segments; s++) {
                const baseFlow = 400 + s * 100;
                const baseSpeed = 40 + s * 5;
                const baseOccupancy = 40 + s * 8;
                
                const timeVariation = Math.sin(t * Math.PI / 30) * 0.2 + 1;
                
                flowRow.push(baseFlow * timeVariation + Math.random() * 50 - 25);
                speedRow.push(baseSpeed / timeVariation + Math.random() * 5 - 2.5);
                occupancyRow.push(baseOccupancy * timeVariation + Math.random() * 10 - 5);
            }
            
            flows.push(flowRow);
            speeds.push(speedRow);
            occupancies.push(occupancyRow);
        }
        
        return {
            timestamp: new Date().toISOString(),
            prediction_horizon_minutes: timeSteps,
            predictions: { flows, speeds, occupancies },
            confidence: 0.82
        };
    }

    generateMockNetwork() {
        return {
            nodes: [
                { id: "n1", name: "Times Square", type: "intersection", lat: 40.7589, lon: -73.9851 },
                { id: "n2", name: "Herald Square", type: "intersection", lat: 40.7505, lon: -73.9934 },
                { id: "n3", name: "Columbus Circle", type: "intersection", lat: 40.7681, lon: -73.9819 },
                { id: "n4", name: "Grand Central", type: "intersection", lat: 40.7527, lon: -73.9772 },
                { id: "n5", name: "Union Square", type: "intersection", lat: 40.7359, lon: -73.9911 }
            ],
            edges: [
                { from: "n1", to: "n2", name: "Broadway", lanes: 3, length: 1.2 },
                { from: "n2", to: "n3", name: "8th Avenue", lanes: 2, length: 2.1 },
                { from: "n1", to: "n4", name: "42nd Street", lanes: 4, length: 0.8 },
                { from: "n4", to: "n5", name: "Park Avenue", lanes: 2, length: 1.5 },
                { from: "n2", to: "n5", name: "6th Avenue", lanes: 3, length: 1.8 }
            ]
        };
    }

    generateMockPerformance() {
        const hours = 24;
        const timestamps = [];
        const metrics = {
            average_speed: [],
            total_flow: [],
            congestion_level: [],
            efficiency: []
        };
        
        const baseTime = new Date();
        baseTime.setHours(baseTime.getHours() - hours);
        
        for (let h = 0; h < hours; h++) {
            const timestamp = new Date(baseTime.getTime() + h * 60 * 60 * 1000);
            timestamps.push(timestamp.toISOString());
            
            const hour = h % 24;
            let speedFactor, flowFactor, congestionFactor;
            
            if ((hour >= 7 && hour <= 9) || (hour >= 17 && hour <= 19)) {
                speedFactor = 0.6 + Math.random() * 0.2;
                flowFactor = 0.8 + Math.random() * 0.2;
                congestionFactor = 0.7 + Math.random() * 0.2;
            } else {
                speedFactor = 0.8 + Math.random() * 0.2;
                flowFactor = 0.4 + Math.random() * 0.3;
                congestionFactor = 0.2 + Math.random() * 0.3;
            }
            
            metrics.average_speed.push(70 * speedFactor);
            metrics.total_flow.push(1000 * flowFactor);
            metrics.congestion_level.push(congestionFactor);
            metrics.efficiency.push(speedFactor * (1 - congestionFactor));
        }
        
        return {
            timestamps,
            metrics,
            summary: {
                avg_speed_24h: metrics.average_speed.reduce((a, b) => a + b) / metrics.average_speed.length,
                total_flow_24h: metrics.total_flow.reduce((a, b) => a + b),
                peak_congestion: Math.max(...metrics.congestion_level),
                avg_efficiency: metrics.efficiency.reduce((a, b) => a + b) / metrics.efficiency.length
            }
        };
    }

    async loadDashboardData() {
        if (window.dashboardManager) {
            await window.dashboardManager.updateDashboard(this.data);
        }
    }

    async loadNetworkData() {
        if (window.networkManager) {
            await window.networkManager.updateNetwork(this.data);
        }
    }

    async loadPredictionData() {
        if (window.predictionsManager) {
            await window.predictionsManager.updatePredictions(this.data);
        }
    }

    async loadAnalyticsData() {
        if (window.analyticsManager) {
            await window.analyticsManager.updateAnalytics(this.data);
        }
    }

    loadFallbackData() {
        this.data.currentStatus = this.generateMockCurrentStatus();
        this.data.predictions = this.generateMockPredictions();
        this.data.network = this.generateMockNetwork();
        this.data.performance = this.generateMockPerformance();
    }

    updateLastUpdatedTime() {
        const lastUpdatedElement = document.getElementById('last-updated');
        if (lastUpdatedElement) {
            const now = new Date();
            lastUpdatedElement.textContent = now.toLocaleTimeString();
        }
    }

    startAutoRefresh() {
        // Refresh data every 30 seconds
        this.updateInterval = setInterval(async () => {
            try {
                await this.refreshData();
            } catch (error) {
                console.error('Auto-refresh failed:', error);
            }
        }, 30000);
    }

    async refreshData() {
        const newData = await this.fetchData('data/current_status.json');
        if (newData && newData.timestamp) {
            this.data.currentStatus = newData;
            this.updateLastUpdatedTime();
            
            // Update current section
            await this.loadSectionData(this.currentSection);
        }
    }

    async refreshCurrentSection() {
        this.showLoading();
        await this.loadSectionData(this.currentSection);
        this.hideLoading();
        this.showNotification('Data refreshed successfully', 'success');
    }

    showLoading() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.classList.add('show');
        }
    }

    hideLoading() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.classList.remove('show');
        }
    }

    showNotification(message, type = 'info') {
        // Create notification element if it doesn't exist
        let notification = document.getElementById('notification');
        if (!notification) {
            notification = document.createElement('div');
            notification.id = 'notification';
            notification.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 1rem 1.5rem;
                border-radius: 5px;
                color: white;
                font-weight: 500;
                z-index: 10000;
                transform: translateX(400px);
                transition: transform 0.3s ease;
            `;
            document.body.appendChild(notification);
        }

        // Set notification style based on type
        const colors = {
            success: '#27ae60',
            warning: '#f39c12',
            error: '#e74c3c',
            info: '#3498db'
        };

        notification.style.backgroundColor = colors[type] || colors.info;
        notification.textContent = message;
        notification.style.transform = 'translateX(0)';

        // Auto-hide after 3 seconds
        setTimeout(() => {
            notification.style.transform = 'translateX(400px)';
        }, 3000);
    }

    // Utility methods
    formatNumber(num, decimals = 0) {
        if (num === null || num === undefined) return '--';
        return Number(num).toLocaleString(undefined, {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        });
    }

    formatPercent(num, decimals = 1) {
        if (num === null || num === undefined) return '--';
        return `${Number(num).toFixed(decimals)}%`;
    }

    formatTime(timestamp) {
        if (!timestamp) return '--';
        return new Date(timestamp).toLocaleTimeString();
    }

    getStatusColor(status) {
        const colors = {
            normal: '#27ae60',
            warning: '#f39c12',
            congested: '#e74c3c',
            error: '#e74c3c'
        };
        return colors[status] || colors.normal;
    }

    destroy() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
    }
}

// Global utility functions
window.refreshNetwork = function() {
    if (window.app) {
        window.app.refreshCurrentSection();
    }
};

window.updatePredictions = function() {
    if (window.predictionsManager) {
        window.predictionsManager.generateNewPredictions();
    }
};

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new TrafficFlowOptApp();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // Page is hidden, pause auto-refresh
        if (window.app && window.app.updateInterval) {
            clearInterval(window.app.updateInterval);
        }
    } else {
        // Page is visible, resume auto-refresh
        if (window.app) {
            window.app.startAutoRefresh();
        }
    }
});

// Handle window resize
window.addEventListener('resize', () => {
    // Trigger chart resizes if needed
    if (window.app) {
        setTimeout(() => {
            window.app.loadSectionData(window.app.currentSection);
        }, 100);
    }
});