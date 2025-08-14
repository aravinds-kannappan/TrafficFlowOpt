#include "include/traffic_network.hpp"
#include "include/traffic_optimizer.hpp"
#include <iostream>
#include <memory>
#include <iomanip>

int main() {
    std::cout << "TrafficFlowOpt - C++20 Traffic Optimization System\n";
    std::cout << "================================================\n\n";
    
    try {
        // Create traffic network
        auto network = std::make_shared<traffic::TrafficNetwork>();
        
        // Add sample road segments
        std::cout << "Setting up traffic network...\n";
        
        traffic::Coordinate coord1(40.7589, -73.9851);  // NYC coordinates
        traffic::Coordinate coord2(40.7614, -73.9776);
        traffic::Coordinate coord3(40.7505, -73.9934);
        
        traffic::RoadSegment segment1("broadway_42nd", coord1, coord2, 3);
        segment1.length_km = 1.2;
        segment1.speed_limit = 35.0;
        segment1.current_flow = 1200.0;
        
        traffic::RoadSegment segment2("7th_ave_42nd", coord2, coord3, 2);
        segment2.length_km = 0.8;
        segment2.speed_limit = 30.0;
        segment2.current_flow = 800.0;
        
        network->add_road_segment(segment1);
        network->add_road_segment(segment2);
        network->connect_segments("broadway_42nd", "7th_ave_42nd");
        
        // Add traffic sensors
        traffic::TrafficSensor sensor1("sensor_001", coord1);
        sensor1.flow_rate = 20.0;  // vehicles per minute
        sensor1.average_speed = 25.0;  // km/h
        sensor1.occupancy = 0.65;  // 65%
        
        network->add_sensor(sensor1);
        
        // Add traffic lights
        traffic::TrafficLight light1("intersection_001", coord2);
        light1.green_time = std::chrono::seconds(45);
        light1.red_time = std::chrono::seconds(35);
        light1.is_green = true;
        
        network->add_traffic_light(light1);
        
        std::cout << "✓ Network setup complete\n\n";
        
        // Validate network
        if (network->validate_network()) {
            std::cout << "✓ Network validation passed\n";
        } else {
            std::cout << "⚠ Network validation warnings:\n";
            auto errors = network->get_validation_errors();
            for (const auto& error : errors) {
                std::cout << "  - " << error << "\n";
            }
        }
        
        // Analyze network
        std::cout << "\nNetwork Analysis:\n";
        std::cout << "================\n";
        
        double efficiency = network->calculate_network_efficiency();
        std::cout << "Network efficiency: " << std::fixed << std::setprecision(2) 
                  << efficiency * 100 << "%\n";
        
        auto bottlenecks = network->find_bottlenecks(0.7);
        if (!bottlenecks.empty()) {
            std::cout << "Bottlenecks detected:\n";
            for (const auto& bottleneck : bottlenecks) {
                std::cout << "  - " << bottleneck << "\n";
            }
        } else {
            std::cout << "No bottlenecks detected\n";
        }
        
        // Route finding
        std::cout << "\nRoute Optimization:\n";
        std::cout << "==================\n";
        
        auto shortest_route = network->find_shortest_path("broadway_42nd", "7th_ave_42nd");
        if (!shortest_route.segment_ids.empty()) {
            std::cout << "Shortest route found:\n";
            for (const auto& segment : shortest_route.segment_ids) {
                std::cout << "  → " << segment << "\n";
            }
            std::cout << "Total distance: " << shortest_route.total_distance << " km\n";
        }
        
        auto fastest_route = network->find_fastest_route("broadway_42nd", "7th_ave_42nd");
        if (!fastest_route.segment_ids.empty()) {
            std::cout << "Fastest route estimated time: " 
                      << fastest_route.estimated_time << " minutes\n";
        }
        
        // Traffic optimization
        std::cout << "\nTraffic Optimization:\n";
        std::cout << "====================\n";
        
        auto optimizer = std::make_unique<traffic::TrafficOptimizer>(network);
        
        // Optimize signal timings
        auto signal_result = optimizer->optimize_signal_timings(
            traffic::OptimizationObjective::MINIMIZE_TRAVEL_TIME);
        
        std::cout << "Signal timing optimization:\n";
        std::cout << "  Objective value: " << signal_result.objective_value << "\n";
        std::cout << "  Converged: " << (signal_result.converged ? "Yes" : "No") << "\n";
        
        if (!signal_result.signal_timings.empty()) {
            std::cout << "  Optimal timings:\n";
            for (const auto& [intersection_id, timing] : signal_result.signal_timings) {
                std::cout << "    " << intersection_id << ": " 
                          << timing.count() << " seconds\n";
            }
        }
        
        // Flow optimization
        auto flow_result = optimizer->optimize_flow_distribution(
            traffic::OptimizationObjective::MINIMIZE_CONGESTION);
        
        std::cout << "\nFlow optimization:\n";
        std::cout << "  Congestion objective: " << flow_result.objective_value << "\n";
        
        // Lane allocation optimization
        std::vector<std::string> segments = {"broadway_42nd", "7th_ave_42nd"};
        auto lane_allocation = optimizer->optimize_lane_allocation(segments);
        
        std::cout << "\nOptimal lane allocation:\n";
        for (const auto& [segment_id, lanes] : lane_allocation) {
            std::cout << "  " << segment_id << ": " << lanes << " lanes\n";
        }
        
        // Performance metrics
        std::cout << "\nPerformance Metrics:\n";
        std::cout << "===================\n";
        
        double travel_time = optimizer->evaluate_objective(
            traffic::OptimizationObjective::MINIMIZE_TRAVEL_TIME);
        double congestion = optimizer->evaluate_objective(
            traffic::OptimizationObjective::MINIMIZE_CONGESTION);
        double throughput = -optimizer->evaluate_objective(
            traffic::OptimizationObjective::MAXIMIZE_THROUGHPUT);
        
        std::cout << "Total travel time: " << travel_time << " minutes\n";
        std::cout << "Total congestion: " << congestion << "\n";
        std::cout << "Total throughput: " << throughput << " vehicles/hour\n";
        
        std::cout << "\n✓ Traffic optimization complete!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\nFor web visualization, open docs/index.html in your browser.\n";
    return 0;
}