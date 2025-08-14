/**
 * Network Manager for TrafficFlowOpt
 * Handles traffic network visualization and mapping
 */

class NetworkManager {
    constructor() {
        this.map = null;
        this.markers = {};
        this.polylines = {};
        this.isInitialized = false;
    }

    async updateNetwork(data) {
        try {
            if (!this.isInitialized) {
                this.initializeMap();
            }
            
            this.updateNetworkMap(data.network, data.currentStatus);
            this.updateSegmentDetails(data.currentStatus);
        } catch (error) {
            console.error('Network update failed:', error);
        }
    }

    initializeMap() {
        const mapContainer = document.getElementById('network-map');
        if (!mapContainer || this.isInitialized) return;

        // Initialize Leaflet map centered on NYC
        this.map = L.map('network-map').setView([40.7589, -73.9851], 13);

        // Add OpenStreetMap tiles
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(this.map);

        this.isInitialized = true;
    }

    updateNetworkMap(networkData, currentStatus) {
        if (!this.map || !networkData) {
            this.createSampleNetwork();
            return;
        }

        // Clear existing markers and polylines
        this.clearMapElements();

        // Add nodes (intersections)
        networkData.nodes.forEach(node => {
            this.addNodeToMap(node, currentStatus);
        });

        // Add edges (roads)
        networkData.edges.forEach(edge => {
            this.addEdgeToMap(edge, networkData.nodes, currentStatus);
        });

        // Fit map to show all markers
        if (Object.keys(this.markers).length > 0) {
            const group = new L.featureGroup(Object.values(this.markers));
            this.map.fitBounds(group.getBounds().pad(0.1));
        }
    }

    createSampleNetwork() {
        if (!this.map) return;

        // Clear existing elements
        this.clearMapElements();

        // Sample NYC intersections
        const sampleNodes = [
            { id: 'n1', name: 'Times Square', lat: 40.7589, lon: -73.9851, status: 'congested' },
            { id: 'n2', name: 'Herald Square', lat: 40.7505, lon: -73.9934, status: 'normal' },
            { id: 'n3', name: 'Columbus Circle', lat: 40.7681, lon: -73.9819, status: 'normal' },
            { id: 'n4', name: 'Grand Central', lat: 40.7527, lon: -73.9772, status: 'warning' },
            { id: 'n5', name: 'Union Square', lat: 40.7359, lon: -73.9911, status: 'normal' }
        ];

        const sampleEdges = [
            { from: 'n1', to: 'n2', name: 'Broadway', flow: 720, status: 'congested' },
            { from: 'n2', to: 'n3', name: '8th Avenue', flow: 450, status: 'normal' },
            { from: 'n1', to: 'n4', name: '42nd Street', flow: 890, status: 'warning' },
            { from: 'n4', to: 'n5', name: 'Park Avenue', flow: 520, status: 'normal' },
            { from: 'n2', to: 'n5', name: '6th Avenue', flow: 640, status: 'warning' }
        ];

        // Add nodes
        sampleNodes.forEach(node => {
            const color = this.getStatusColor(node.status);
            const marker = L.circleMarker([node.lat, node.lon], {
                radius: 8,
                fillColor: color,
                color: '#ffffff',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.8
            }).addTo(this.map);

            marker.bindPopup(`
                <div class="map-popup">
                    <h4>${node.name}</h4>
                    <p><strong>Status:</strong> ${node.status}</p>
                    <p><strong>Type:</strong> Intersection</p>
                </div>
            `);

            this.markers[node.id] = marker;
        });

        // Add edges
        sampleEdges.forEach(edge => {
            const fromNode = sampleNodes.find(n => n.id === edge.from);
            const toNode = sampleNodes.find(n => n.id === edge.to);
            
            if (fromNode && toNode) {
                const color = this.getStatusColor(edge.status);
                const weight = Math.max(3, Math.min(edge.flow / 200, 10));
                
                const polyline = L.polyline([
                    [fromNode.lat, fromNode.lon],
                    [toNode.lat, toNode.lon]
                ], {
                    color: color,
                    weight: weight,
                    opacity: 0.8
                }).addTo(this.map);

                polyline.bindPopup(`
                    <div class="map-popup">
                        <h4>${edge.name}</h4>
                        <p><strong>Flow:</strong> ${edge.flow} veh/h</p>
                        <p><strong>Status:</strong> ${edge.status}</p>
                        <p><strong>From:</strong> ${fromNode.name}</p>
                        <p><strong>To:</strong> ${toNode.name}</p>
                    </div>
                `);

                this.polylines[`${edge.from}-${edge.to}`] = polyline;
            }
        });

        // Fit map to show all elements
        const group = new L.featureGroup([...Object.values(this.markers), ...Object.values(this.polylines)]);
        this.map.fitBounds(group.getBounds().pad(0.1));
    }

    addNodeToMap(node, currentStatus) {
        const status = this.getNodeStatus(node, currentStatus);
        const color = this.getStatusColor(status);
        
        const marker = L.circleMarker([node.lat, node.lon], {
            radius: 10,
            fillColor: color,
            color: '#ffffff',
            weight: 2,
            opacity: 1,
            fillOpacity: 0.8
        }).addTo(this.map);

        // Create popup content
        const popupContent = this.createNodePopup(node, status, currentStatus);
        marker.bindPopup(popupContent);

        this.markers[node.id] = marker;
    }

    addEdgeToMap(edge, nodes, currentStatus) {
        const fromNode = nodes.find(n => n.id === edge.from);
        const toNode = nodes.find(n => n.id === edge.to);
        
        if (!fromNode || !toNode) return;

        const edgeStatus = this.getEdgeStatus(edge, currentStatus);
        const color = this.getStatusColor(edgeStatus.status);
        const weight = Math.max(3, Math.min(edgeStatus.flow / 200, 10));
        
        const polyline = L.polyline([
            [fromNode.lat, fromNode.lon],
            [toNode.lat, toNode.lon]
        ], {
            color: color,
            weight: weight,
            opacity: 0.8
        }).addTo(this.map);

        // Create popup content
        const popupContent = this.createEdgePopup(edge, edgeStatus, fromNode, toNode);
        polyline.bindPopup(popupContent);

        this.polylines[`${edge.from}-${edge.to}`] = polyline;
    }

    getNodeStatus(node, currentStatus) {
        if (!currentStatus || !currentStatus.segments) return 'normal';
        
        // Find segments near this node
        const nearbySegments = currentStatus.segments.filter(segment => {
            const distance = this.calculateDistance(
                node.lat, node.lon,
                segment.latitude, segment.longitude
            );
            return distance < 0.01; // Within ~1km
        });

        if (nearbySegments.length === 0) return 'normal';

        // Determine status based on nearby segments
        const congestedCount = nearbySegments.filter(s => s.status === 'congested').length;
        if (congestedCount > nearbySegments.length / 2) return 'congested';
        if (congestedCount > 0) return 'warning';
        return 'normal';
    }

    getEdgeStatus(edge, currentStatus) {
        if (!currentStatus || !currentStatus.segments) {
            return { status: 'normal', flow: 500, speed: 40 };
        }

        // Try to match edge with a segment
        const matchingSegment = currentStatus.segments.find(segment => {
            return segment.name.toLowerCase().includes(edge.name.toLowerCase()) ||
                   edge.name.toLowerCase().includes(segment.name.toLowerCase());
        });

        if (matchingSegment) {
            return {
                status: matchingSegment.status,
                flow: matchingSegment.flow_rate,
                speed: matchingSegment.average_speed
            };
        }

        // Default values
        return { status: 'normal', flow: 500, speed: 40 };
    }

    createNodePopup(node, status, currentStatus) {
        return `
            <div class="map-popup">
                <h4>${node.name}</h4>
                <p><strong>Type:</strong> ${node.type || 'Intersection'}</p>
                <p><strong>Status:</strong> <span style="color: ${this.getStatusColor(status)}">${status}</span></p>
                <p><strong>Coordinates:</strong> ${node.lat.toFixed(4)}, ${node.lon.toFixed(4)}</p>
            </div>
        `;
    }

    createEdgePopup(edge, edgeStatus, fromNode, toNode) {
        return `
            <div class="map-popup">
                <h4>${edge.name}</h4>
                <p><strong>Flow:</strong> ${Math.round(edgeStatus.flow)} veh/h</p>
                <p><strong>Speed:</strong> ${edgeStatus.speed.toFixed(1)} km/h</p>
                <p><strong>Status:</strong> <span style="color: ${this.getStatusColor(edgeStatus.status)}">${edgeStatus.status}</span></p>
                <p><strong>Lanes:</strong> ${edge.lanes || 'N/A'}</p>
                <p><strong>Length:</strong> ${edge.length ? edge.length.toFixed(1) + ' km' : 'N/A'}</p>
                <hr>
                <p><strong>From:</strong> ${fromNode.name}</p>
                <p><strong>To:</strong> ${toNode.name}</p>
            </div>
        `;
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

    calculateDistance(lat1, lon1, lat2, lon2) {
        const R = 6371; // Earth's radius in km
        const dLat = (lat2 - lat1) * Math.PI / 180;
        const dLon = (lon2 - lon1) * Math.PI / 180;
        const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
                  Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
                  Math.sin(dLon/2) * Math.sin(dLon/2);
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
        return R * c;
    }

    updateSegmentDetails(currentStatus) {
        const container = document.getElementById('segment-grid');
        if (!container) return;

        if (!currentStatus || !currentStatus.segments) {
            this.createSampleSegmentDetails(container);
            return;
        }

        container.innerHTML = '';

        currentStatus.segments.forEach(segment => {
            const segmentItem = document.createElement('div');
            segmentItem.className = 'segment-item';
            
            segmentItem.innerHTML = `
                <div class="segment-name">${segment.name}</div>
                <div class="segment-metrics">
                    <div><strong>Flow:</strong> ${Math.round(segment.flow_rate)} veh/h</div>
                    <div><strong>Speed:</strong> ${segment.average_speed.toFixed(1)} km/h</div>
                    <div><strong>Occupancy:</strong> ${Math.round(segment.occupancy)}%</div>
                </div>
                <div class="segment-status" style="color: ${this.getStatusColor(segment.status)}">
                    Status: ${segment.status.toUpperCase()}
                </div>
            `;
            
            container.appendChild(segmentItem);
        });
    }

    createSampleSegmentDetails(container) {
        const sampleSegments = [
            {
                name: 'Broadway 42nd Street',
                flow_rate: 720,
                average_speed: 38.5,
                occupancy: 68,
                status: 'congested'
            },
            {
                name: '7th Avenue 42nd Street',
                flow_rate: 580,
                average_speed: 52.1,
                occupancy: 55,
                status: 'normal'
            },
            {
                name: 'Times Square North',
                flow_rate: 950,
                average_speed: 25.3,
                occupancy: 85,
                status: 'congested'
            },
            {
                name: 'Herald Square',
                flow_rate: 420,
                average_speed: 45.8,
                occupancy: 42,
                status: 'normal'
            },
            {
                name: '8th Avenue Corridor',
                flow_rate: 650,
                average_speed: 35.2,
                occupancy: 72,
                status: 'warning'
            }
        ];

        container.innerHTML = '';

        sampleSegments.forEach(segment => {
            const segmentItem = document.createElement('div');
            segmentItem.className = 'segment-item';
            
            segmentItem.innerHTML = `
                <div class="segment-name">${segment.name}</div>
                <div class="segment-metrics">
                    <div><strong>Flow:</strong> ${Math.round(segment.flow_rate)} veh/h</div>
                    <div><strong>Speed:</strong> ${segment.average_speed.toFixed(1)} km/h</div>
                    <div><strong>Occupancy:</strong> ${Math.round(segment.occupancy)}%</div>
                </div>
                <div class="segment-status" style="color: ${this.getStatusColor(segment.status)}">
                    Status: ${segment.status.toUpperCase()}
                </div>
            `;
            
            container.appendChild(segmentItem);
        });
    }

    clearMapElements() {
        // Clear markers
        Object.values(this.markers).forEach(marker => {
            this.map.removeLayer(marker);
        });
        this.markers = {};

        // Clear polylines
        Object.values(this.polylines).forEach(polyline => {
            this.map.removeLayer(polyline);
        });
        this.polylines = {};
    }

    refreshNetwork() {
        if (window.app && window.app.data) {
            this.updateNetwork(window.app.data);
        }
    }

    destroy() {
        if (this.map) {
            this.clearMapElements();
            this.map.remove();
            this.map = null;
        }
        this.isInitialized = false;
    }
}

// Initialize network manager
window.networkManager = new NetworkManager();