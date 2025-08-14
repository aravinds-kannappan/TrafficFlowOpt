#pragma once

#include <vector>
#include <string>
#include <chrono>
#include <unordered_map>
#include <memory>

namespace traffic {

using TimePoint = std::chrono::system_clock::time_point;
using Duration = std::chrono::seconds;

struct Coordinate {
    double latitude;
    double longitude;
    
    Coordinate(double lat = 0.0, double lon = 0.0) 
        : latitude(lat), longitude(lon) {}
};

struct TrafficSensor {
    std::string id;
    Coordinate location;
    double flow_rate;        // vehicles per minute
    double average_speed;    // km/h
    double occupancy;        // percentage
    TimePoint timestamp;
    
    TrafficSensor(const std::string& sensor_id, const Coordinate& coord)
        : id(sensor_id), location(coord), flow_rate(0.0), 
          average_speed(0.0), occupancy(0.0) {}
};

struct TrafficLight {
    std::string intersection_id;
    Coordinate location;
    Duration green_time;
    Duration red_time;
    Duration yellow_time;
    TimePoint last_change;
    bool is_green;
    
    TrafficLight(const std::string& id, const Coordinate& coord)
        : intersection_id(id), location(coord), 
          green_time(30), red_time(30), yellow_time(3),
          is_green(false) {}
};

struct RoadSegment {
    std::string id;
    Coordinate start;
    Coordinate end;
    int lanes;
    double length_km;
    double speed_limit;
    double capacity;         // vehicles per hour
    double current_flow;
    
    RoadSegment(const std::string& segment_id, const Coordinate& start_coord,
                const Coordinate& end_coord, int num_lanes)
        : id(segment_id), start(start_coord), end(end_coord), 
          lanes(num_lanes), length_km(0.0), speed_limit(50.0),
          capacity(1800.0 * num_lanes), current_flow(0.0) {}
};

struct Route {
    std::vector<std::string> segment_ids;
    double total_distance;
    double estimated_time;
    double congestion_factor;
    
    Route() : total_distance(0.0), estimated_time(0.0), congestion_factor(1.0) {}
};

struct TrafficMatrix {
    std::vector<std::vector<double>> flow_matrix;
    std::vector<std::string> node_ids;
    size_t size;
    
    TrafficMatrix(size_t n) : size(n) {
        flow_matrix.resize(n, std::vector<double>(n, 0.0));
        node_ids.resize(n);
    }
};

enum class OptimizationObjective {
    MINIMIZE_TRAVEL_TIME,
    MINIMIZE_CONGESTION,
    MAXIMIZE_THROUGHPUT,
    BALANCE_FLOW
};

struct OptimizationResult {
    std::unordered_map<std::string, Duration> signal_timings;
    std::unordered_map<std::string, Route> optimal_routes;
    double objective_value;
    bool converged;
    
    OptimizationResult() : objective_value(0.0), converged(false) {}
};

} // namespace traffic