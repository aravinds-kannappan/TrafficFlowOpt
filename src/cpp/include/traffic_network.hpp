#pragma once

#include "traffic_types.hpp"
#include <memory>
#include <vector>
#include <unordered_map>
#include <functional>

namespace traffic {

class TrafficNetwork {
public:
    TrafficNetwork();
    ~TrafficNetwork() = default;

    // Network building
    void add_sensor(const TrafficSensor& sensor);
    void add_traffic_light(const TrafficLight& light);
    void add_road_segment(const RoadSegment& segment);
    void connect_segments(const std::string& from_id, const std::string& to_id);

    // Data updates
    void update_sensor_data(const std::string& sensor_id, double flow, 
                           double speed, double occupancy);
    void update_traffic_light(const std::string& light_id, bool is_green);

    // Network analysis
    TrafficMatrix compute_flow_matrix() const;
    std::vector<std::string> find_bottlenecks(double threshold = 0.8) const;
    double calculate_network_efficiency() const;

    // Route operations
    Route find_shortest_path(const std::string& start, const std::string& end) const;
    Route find_fastest_route(const std::string& start, const std::string& end) const;
    std::vector<Route> find_alternative_routes(const std::string& start, 
                                              const std::string& end, int num_routes = 3) const;

    // Getters
    const std::unordered_map<std::string, TrafficSensor>& get_sensors() const { return sensors_; }
    const std::unordered_map<std::string, TrafficLight>& get_traffic_lights() const { return traffic_lights_; }
    const std::unordered_map<std::string, RoadSegment>& get_road_segments() const { return road_segments_; }
    const std::unordered_map<std::string, std::vector<std::string>>& get_adjacency_list() const { return adjacency_list_; }

    // Validation
    bool validate_network() const;
    std::vector<std::string> get_validation_errors() const;

private:
    std::unordered_map<std::string, TrafficSensor> sensors_;
    std::unordered_map<std::string, TrafficLight> traffic_lights_;
    std::unordered_map<std::string, RoadSegment> road_segments_;
    std::unordered_map<std::string, std::vector<std::string>> adjacency_list_;

    // Helper methods
    double calculate_segment_travel_time(const std::string& segment_id) const;
    double calculate_congestion_factor(const std::string& segment_id) const;
    bool is_valid_path(const std::vector<std::string>& path) const;
};

} // namespace traffic