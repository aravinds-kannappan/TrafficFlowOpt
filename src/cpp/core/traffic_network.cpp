#include "../include/traffic_network.hpp"
#include <algorithm>
#include <queue>
#include <limits>
#include <cmath>

namespace traffic {

TrafficNetwork::TrafficNetwork() {
}

void TrafficNetwork::add_sensor(const TrafficSensor& sensor) {
    sensors_[sensor.id] = sensor;
}

void TrafficNetwork::add_traffic_light(const TrafficLight& light) {
    traffic_lights_[light.intersection_id] = light;
}

void TrafficNetwork::add_road_segment(const RoadSegment& segment) {
    road_segments_[segment.id] = segment;
}

void TrafficNetwork::connect_segments(const std::string& from_id, const std::string& to_id) {
    adjacency_list_[from_id].push_back(to_id);
}

void TrafficNetwork::update_sensor_data(const std::string& sensor_id, double flow, 
                                       double speed, double occupancy) {
    auto it = sensors_.find(sensor_id);
    if (it != sensors_.end()) {
        it->second.flow_rate = flow;
        it->second.average_speed = speed;
        it->second.occupancy = occupancy;
        it->second.timestamp = std::chrono::system_clock::now();
    }
}

void TrafficNetwork::update_traffic_light(const std::string& light_id, bool is_green) {
    auto it = traffic_lights_.find(light_id);
    if (it != traffic_lights_.end()) {
        it->second.is_green = is_green;
        it->second.last_change = std::chrono::system_clock::now();
    }
}

TrafficMatrix TrafficNetwork::compute_flow_matrix() const {
    size_t n = road_segments_.size();
    TrafficMatrix matrix(n);
    
    size_t i = 0;
    for (const auto& [id, segment] : road_segments_) {
        matrix.node_ids[i] = id;
        ++i;
    }
    
    // Fill flow matrix based on current traffic flows
    for (i = 0; i < n; ++i) {
        const std::string& from_id = matrix.node_ids[i];
        auto adj_it = adjacency_list_.find(from_id);
        if (adj_it != adjacency_list_.end()) {
            for (const std::string& to_id : adj_it->second) {
                auto to_it = std::find(matrix.node_ids.begin(), matrix.node_ids.end(), to_id);
                if (to_it != matrix.node_ids.end()) {
                    size_t j = std::distance(matrix.node_ids.begin(), to_it);
                    auto segment_it = road_segments_.find(from_id);
                    if (segment_it != road_segments_.end()) {
                        matrix.flow_matrix[i][j] = segment_it->second.current_flow;
                    }
                }
            }
        }
    }
    
    return matrix;
}

std::vector<std::string> TrafficNetwork::find_bottlenecks(double threshold) const {
    std::vector<std::string> bottlenecks;
    
    for (const auto& [id, segment] : road_segments_) {
        double utilization = segment.current_flow / segment.capacity;
        if (utilization >= threshold) {
            bottlenecks.push_back(id);
        }
    }
    
    return bottlenecks;
}

double TrafficNetwork::calculate_network_efficiency() const {
    if (road_segments_.empty()) return 0.0;
    
    double total_efficiency = 0.0;
    
    for (const auto& [id, segment] : road_segments_) {
        double utilization = segment.current_flow / segment.capacity;
        // Efficiency drops off as utilization approaches 1.0
        double efficiency = utilization * (1.0 - std::pow(utilization, 4));
        total_efficiency += efficiency;
    }
    
    return total_efficiency / road_segments_.size();
}

Route TrafficNetwork::find_shortest_path(const std::string& start, const std::string& end) const {
    // Dijkstra's algorithm for shortest path by distance
    std::unordered_map<std::string, double> distances;
    std::unordered_map<std::string, std::string> previous;
    std::priority_queue<std::pair<double, std::string>, 
                       std::vector<std::pair<double, std::string>>,
                       std::greater<>> pq;
    
    // Initialize distances
    for (const auto& [id, segment] : road_segments_) {
        distances[id] = std::numeric_limits<double>::infinity();
    }
    distances[start] = 0.0;
    pq.push({0.0, start});
    
    while (!pq.empty()) {
        auto [dist, current] = pq.top();
        pq.pop();
        
        if (current == end) break;
        if (dist > distances[current]) continue;
        
        auto adj_it = adjacency_list_.find(current);
        if (adj_it != adjacency_list_.end()) {
            for (const std::string& neighbor : adj_it->second) {
                auto segment_it = road_segments_.find(current);
                if (segment_it != road_segments_.end()) {
                    double edge_weight = segment_it->second.length_km;
                    double new_dist = distances[current] + edge_weight;
                    
                    if (new_dist < distances[neighbor]) {
                        distances[neighbor] = new_dist;
                        previous[neighbor] = current;
                        pq.push({new_dist, neighbor});
                    }
                }
            }
        }
    }
    
    // Reconstruct path
    Route route;
    if (distances[end] != std::numeric_limits<double>::infinity()) {
        std::string current = end;
        while (current != start && previous.find(current) != previous.end()) {
            route.segment_ids.push_back(current);
            current = previous[current];
        }
        route.segment_ids.push_back(start);
        std::reverse(route.segment_ids.begin(), route.segment_ids.end());
        route.total_distance = distances[end];
    }
    
    return route;
}

Route TrafficNetwork::find_fastest_route(const std::string& start, const std::string& end) const {
    // Similar to shortest path but using travel time as weight
    std::unordered_map<std::string, double> times;
    std::unordered_map<std::string, std::string> previous;
    std::priority_queue<std::pair<double, std::string>, 
                       std::vector<std::pair<double, std::string>>,
                       std::greater<>> pq;
    
    for (const auto& [id, segment] : road_segments_) {
        times[id] = std::numeric_limits<double>::infinity();
    }
    times[start] = 0.0;
    pq.push({0.0, start});
    
    while (!pq.empty()) {
        auto [time, current] = pq.top();
        pq.pop();
        
        if (current == end) break;
        if (time > times[current]) continue;
        
        auto adj_it = adjacency_list_.find(current);
        if (adj_it != adjacency_list_.end()) {
            for (const std::string& neighbor : adj_it->second) {
                double travel_time = calculate_segment_travel_time(current);
                double new_time = times[current] + travel_time;
                
                if (new_time < times[neighbor]) {
                    times[neighbor] = new_time;
                    previous[neighbor] = current;
                    pq.push({new_time, neighbor});
                }
            }
        }
    }
    
    // Reconstruct route
    Route route;
    if (times[end] != std::numeric_limits<double>::infinity()) {
        std::string current = end;
        while (current != start && previous.find(current) != previous.end()) {
            route.segment_ids.push_back(current);
            current = previous[current];
        }
        route.segment_ids.push_back(start);
        std::reverse(route.segment_ids.begin(), route.segment_ids.end());
        route.estimated_time = times[end];
    }
    
    return route;
}

std::vector<Route> TrafficNetwork::find_alternative_routes(const std::string& start, 
                                                          const std::string& end, 
                                                          int num_routes) const {
    std::vector<Route> routes;
    
    // Find primary route
    Route primary = find_fastest_route(start, end);
    if (!primary.segment_ids.empty()) {
        routes.push_back(primary);
    }
    
    // Find alternative routes by temporarily removing edges from primary route
    for (int i = 1; i < num_routes && i < static_cast<int>(primary.segment_ids.size()); ++i) {
        // This is a simplified approach - in practice, you'd use more sophisticated algorithms
        // like Yen's k-shortest paths algorithm
        Route alt_route = find_shortest_path(start, end);
        if (!alt_route.segment_ids.empty() && alt_route.segment_ids != primary.segment_ids) {
            routes.push_back(alt_route);
        }
    }
    
    return routes;
}

bool TrafficNetwork::validate_network() const {
    return get_validation_errors().empty();
}

std::vector<std::string> TrafficNetwork::get_validation_errors() const {
    std::vector<std::string> errors;
    
    // Check for isolated segments
    for (const auto& [id, segment] : road_segments_) {
        bool has_incoming = false;
        bool has_outgoing = false;
        
        for (const auto& [from_id, neighbors] : adjacency_list_) {
            if (from_id == id && !neighbors.empty()) {
                has_outgoing = true;
            }
            if (std::find(neighbors.begin(), neighbors.end(), id) != neighbors.end()) {
                has_incoming = true;
            }
        }
        
        if (!has_incoming && !has_outgoing) {
            errors.push_back("Isolated segment: " + id);
        }
    }
    
    // Check for segments with invalid capacity
    for (const auto& [id, segment] : road_segments_) {
        if (segment.capacity <= 0) {
            errors.push_back("Invalid capacity for segment: " + id);
        }
    }
    
    return errors;
}

double TrafficNetwork::calculate_segment_travel_time(const std::string& segment_id) const {
    auto it = road_segments_.find(segment_id);
    if (it == road_segments_.end()) return 0.0;
    
    const RoadSegment& segment = it->second;
    double congestion_factor = calculate_congestion_factor(segment_id);
    
    // Base travel time adjusted for congestion
    double base_time = (segment.length_km / segment.speed_limit) * 60.0; // minutes
    return base_time * congestion_factor;
}

double TrafficNetwork::calculate_congestion_factor(const std::string& segment_id) const {
    auto it = road_segments_.find(segment_id);
    if (it == road_segments_.end()) return 1.0;
    
    const RoadSegment& segment = it->second;
    double utilization = segment.current_flow / segment.capacity;
    
    // BPR (Bureau of Public Roads) function
    double alpha = 0.15;
    double beta = 4.0;
    return 1.0 + alpha * std::pow(utilization, beta);
}

bool TrafficNetwork::is_valid_path(const std::vector<std::string>& path) const {
    if (path.size() < 2) return false;
    
    for (size_t i = 0; i < path.size() - 1; ++i) {
        auto adj_it = adjacency_list_.find(path[i]);
        if (adj_it == adjacency_list_.end()) return false;
        
        const std::vector<std::string>& neighbors = adj_it->second;
        if (std::find(neighbors.begin(), neighbors.end(), path[i + 1]) == neighbors.end()) {
            return false;
        }
    }
    
    return true;
}

} // namespace traffic