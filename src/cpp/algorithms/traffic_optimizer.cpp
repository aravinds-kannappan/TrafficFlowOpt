#include "../include/traffic_optimizer.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <numeric>

namespace traffic {

TrafficOptimizer::TrafficOptimizer(std::shared_ptr<TrafficNetwork> network)
    : network_(network), tolerance_(1e-6), max_iterations_(1000),
      travel_time_weight_(1.0), congestion_weight_(1.0) {
}

OptimizationResult TrafficOptimizer::optimize_signal_timings(
    OptimizationObjective objective, const std::vector<std::string>& intersection_ids) {
    
    std::vector<std::string> target_intersections = intersection_ids;
    if (target_intersections.empty()) {
        // Use all intersections if none specified
        const auto& lights = network_->get_traffic_lights();
        for (const auto& [id, light] : lights) {
            target_intersections.push_back(id);
        }
    }
    
    return gradient_descent_optimization(objective, target_intersections);
}

OptimizationResult TrafficOptimizer::optimize_flow_distribution(OptimizationObjective objective) {
    const auto& segments = network_->get_road_segments();
    std::vector<std::string> segment_ids;
    for (const auto& [id, segment] : segments) {
        segment_ids.push_back(id);
    }
    
    return simulated_annealing_optimization(objective);
}

std::vector<Route> TrafficOptimizer::optimize_routes_for_demand(
    const std::vector<std::pair<std::string, std::string>>& origin_destination_pairs) {
    
    std::vector<Route> optimized_routes;
    
    for (const auto& [origin, destination] : origin_destination_pairs) {
        // Find multiple route options
        std::vector<Route> candidate_routes = network_->find_alternative_routes(origin, destination, 3);
        
        if (!candidate_routes.empty()) {
            // Select best route based on current conditions
            Route best_route = candidate_routes[0];
            double best_score = std::numeric_limits<double>::infinity();
            
            for (const Route& route : candidate_routes) {
                double score = route.estimated_time * (1.0 + route.congestion_factor);
                if (score < best_score) {
                    best_score = score;
                    best_route = route;
                }
            }
            
            optimized_routes.push_back(best_route);
        }
    }
    
    return optimized_routes;
}

std::unordered_map<std::string, int> TrafficOptimizer::optimize_lane_allocation(
    const std::vector<std::string>& segment_ids) {
    
    std::unordered_map<std::string, int> lane_allocation;
    const auto& segments = network_->get_road_segments();
    
    for (const std::string& segment_id : segment_ids) {
        auto it = segments.find(segment_id);
        if (it != segments.end()) {
            const RoadSegment& segment = it->second;
            double utilization = segment.current_flow / segment.capacity;
            
            // Dynamic lane allocation based on utilization
            int optimal_lanes = segment.lanes;
            if (utilization > 0.8) {
                optimal_lanes = std::min(segment.lanes + 1, 4); // Max 4 lanes
            } else if (utilization < 0.3) {
                optimal_lanes = std::max(segment.lanes - 1, 1); // Min 1 lane
            }
            
            lane_allocation[segment_id] = optimal_lanes;
        }
    }
    
    return lane_allocation;
}

OptimizationResult TrafficOptimizer::optimize_for_incident(
    const std::string& incident_segment_id, double capacity_reduction_factor) {
    
    OptimizationResult result;
    
    // Temporarily reduce capacity of incident segment
    auto& segments = const_cast<std::unordered_map<std::string, RoadSegment>&>(
        network_->get_road_segments());
    auto it = segments.find(incident_segment_id);
    if (it != segments.end()) {
        double original_capacity = it->second.capacity;
        it->second.capacity *= capacity_reduction_factor;
        
        // Optimize with reduced capacity
        result = optimize_flow_distribution(OptimizationObjective::MINIMIZE_CONGESTION);
        
        // Restore original capacity
        it->second.capacity = original_capacity;
    }
    
    return result;
}

OptimizationResult TrafficOptimizer::optimize_for_predicted_demand(
    const std::unordered_map<std::string, double>& predicted_flows, int time_horizon_minutes) {
    
    // Store current flows
    const auto& segments = network_->get_road_segments();
    std::unordered_map<std::string, double> original_flows;
    for (const auto& [id, segment] : segments) {
        original_flows[id] = segment.current_flow;
    }
    
    // Apply predicted flows
    auto& mutable_segments = const_cast<std::unordered_map<std::string, RoadSegment>&>(segments);
    for (const auto& [segment_id, predicted_flow] : predicted_flows) {
        auto it = mutable_segments.find(segment_id);
        if (it != mutable_segments.end()) {
            it->second.current_flow = predicted_flow;
        }
    }
    
    // Optimize for predicted conditions
    OptimizationResult result = optimize_signal_timings(OptimizationObjective::MINIMIZE_TRAVEL_TIME);
    
    // Restore original flows
    for (const auto& [segment_id, original_flow] : original_flows) {
        auto it = mutable_segments.find(segment_id);
        if (it != mutable_segments.end()) {
            it->second.current_flow = original_flow;
        }
    }
    
    return result;
}

double TrafficOptimizer::evaluate_objective(OptimizationObjective objective) const {
    switch (objective) {
        case OptimizationObjective::MINIMIZE_TRAVEL_TIME:
            return calculate_total_travel_time();
        case OptimizationObjective::MINIMIZE_CONGESTION:
            return calculate_total_congestion();
        case OptimizationObjective::MAXIMIZE_THROUGHPUT:
            return -calculate_total_throughput(); // Negative for minimization
        case OptimizationObjective::BALANCE_FLOW:
            return calculate_flow_balance();
        default:
            return 0.0;
    }
}

void TrafficOptimizer::set_optimization_parameters(double tolerance, int max_iterations) {
    tolerance_ = tolerance;
    max_iterations_ = max_iterations;
}

void TrafficOptimizer::set_weights(double travel_time_weight, double congestion_weight) {
    travel_time_weight_ = travel_time_weight;
    congestion_weight_ = congestion_weight;
}

OptimizationResult TrafficOptimizer::gradient_descent_optimization(
    OptimizationObjective objective, const std::vector<std::string>& variables) {
    
    OptimizationResult result;
    optimization_history_.clear();
    
    // Initialize with current state
    std::vector<double> x(variables.size());
    const auto& lights = network_->get_traffic_lights();
    for (size_t i = 0; i < variables.size(); ++i) {
        auto it = lights.find(variables[i]);
        if (it != lights.end()) {
            x[i] = std::chrono::duration_cast<std::chrono::seconds>(it->second.green_time).count();
        }
    }
    
    double learning_rate = 0.01;
    
    for (int iteration = 0; iteration < max_iterations_; ++iteration) {
        std::vector<double> gradient = compute_gradient(objective, variables);
        
        // Update variables
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] -= learning_rate * gradient[i];
            x[i] = std::max(10.0, std::min(x[i], 120.0)); // Constrain to [10, 120] seconds
        }
        
        double current_objective = evaluate_objective(objective);
        optimization_history_.push_back(current_objective);
        
        // Check convergence
        if (gradient.size() > 0) {
            double gradient_norm = std::sqrt(std::inner_product(
                gradient.begin(), gradient.end(), gradient.begin(), 0.0));
            if (gradient_norm < tolerance_) {
                result.converged = true;
                break;
            }
        }
    }
    
    // Update signal timings in result
    for (size_t i = 0; i < variables.size(); ++i) {
        result.signal_timings[variables[i]] = Duration(static_cast<int>(x[i]));
    }
    
    result.objective_value = evaluate_objective(objective);
    return result;
}

OptimizationResult TrafficOptimizer::genetic_algorithm_optimization(OptimizationObjective objective) {
    // Simplified genetic algorithm implementation
    OptimizationResult result;
    
    const int population_size = 50;
    const int generations = 100;
    const double mutation_rate = 0.1;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    // Initialize population (simplified)
    std::vector<std::vector<double>> population(population_size);
    const auto& lights = network_->get_traffic_lights();
    
    for (int i = 0; i < population_size; ++i) {
        population[i].resize(lights.size());
        for (size_t j = 0; j < lights.size(); ++j) {
            population[i][j] = 10.0 + dis(gen) * 110.0; // Random timing [10, 120]
        }
    }
    
    for (int generation = 0; generation < generations; ++generation) {
        // Evaluate fitness (simplified)
        std::vector<double> fitness(population_size);
        for (int i = 0; i < population_size; ++i) {
            fitness[i] = -evaluate_objective(objective); // Higher is better
        }
        
        // Selection and crossover (simplified)
        std::vector<std::vector<double>> new_population;
        for (int i = 0; i < population_size; ++i) {
            // Tournament selection
            int parent1 = std::max_element(fitness.begin(), fitness.end()) - fitness.begin();
            int parent2 = std::max_element(fitness.begin(), fitness.end()) - fitness.begin();
            
            // Crossover
            std::vector<double> offspring = population[parent1];
            for (size_t j = 0; j < offspring.size(); ++j) {
                if (dis(gen) < 0.5) {
                    offspring[j] = population[parent2][j];
                }
                
                // Mutation
                if (dis(gen) < mutation_rate) {
                    offspring[j] += (dis(gen) - 0.5) * 20.0; // ±10 seconds
                    offspring[j] = std::max(10.0, std::min(offspring[j], 120.0));
                }
            }
            new_population.push_back(offspring);
        }
        population = new_population;
    }
    
    result.objective_value = evaluate_objective(objective);
    return result;
}

OptimizationResult TrafficOptimizer::simulated_annealing_optimization(OptimizationObjective objective) {
    OptimizationResult result;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    double temperature = 1000.0;
    double cooling_rate = 0.95;
    double current_objective = evaluate_objective(objective);
    double best_objective = current_objective;
    
    for (int iteration = 0; iteration < max_iterations_; ++iteration) {
        // Generate neighbor solution (simplified)
        double delta = (dis(gen) - 0.5) * 10.0; // Small random change
        
        // Evaluate neighbor
        double new_objective = current_objective + delta; // Simplified
        
        // Accept or reject
        double delta_e = new_objective - current_objective;
        if (delta_e < 0 || dis(gen) < std::exp(-delta_e / temperature)) {
            current_objective = new_objective;
            if (current_objective < best_objective) {
                best_objective = current_objective;
            }
        }
        
        temperature *= cooling_rate;
        
        if (temperature < tolerance_) break;
    }
    
    result.objective_value = best_objective;
    result.converged = true;
    return result;
}

double TrafficOptimizer::calculate_total_travel_time() const {
    double total_time = 0.0;
    const auto& segments = network_->get_road_segments();
    
    for (const auto& [id, segment] : segments) {
        if (segment.current_flow > 0) {
            double congestion_factor = std::max(1.0, segment.current_flow / segment.capacity);
            double travel_time = (segment.length_km / segment.speed_limit) * 60.0 * congestion_factor;
            total_time += travel_time * segment.current_flow;
        }
    }
    
    return total_time;
}

double TrafficOptimizer::calculate_total_congestion() const {
    double total_congestion = 0.0;
    const auto& segments = network_->get_road_segments();
    
    for (const auto& [id, segment] : segments) {
        double utilization = segment.current_flow / segment.capacity;
        total_congestion += std::max(0.0, utilization - 0.8); // Congestion above 80% capacity
    }
    
    return total_congestion;
}

double TrafficOptimizer::calculate_total_throughput() const {
    double total_throughput = 0.0;
    const auto& segments = network_->get_road_segments();
    
    for (const auto& [id, segment] : segments) {
        total_throughput += segment.current_flow;
    }
    
    return total_throughput;
}

double TrafficOptimizer::calculate_flow_balance() const {
    double imbalance = 0.0;
    const auto& segments = network_->get_road_segments();
    
    // Calculate variance in utilization
    std::vector<double> utilizations;
    for (const auto& [id, segment] : segments) {
        utilizations.push_back(segment.current_flow / segment.capacity);
    }
    
    if (!utilizations.empty()) {
        double mean = std::accumulate(utilizations.begin(), utilizations.end(), 0.0) / utilizations.size();
        for (double util : utilizations) {
            imbalance += std::pow(util - mean, 2);
        }
        imbalance /= utilizations.size();
    }
    
    return imbalance;
}

std::vector<double> TrafficOptimizer::compute_gradient(
    OptimizationObjective objective, const std::vector<std::string>& variables) const {
    
    std::vector<double> gradient(variables.size());
    const double h = 1e-6; // Small step for numerical differentiation
    
    double base_value = evaluate_objective(objective);
    
    for (size_t i = 0; i < variables.size(); ++i) {
        // Perturb variable and compute partial derivative
        // This is simplified - in practice, you'd modify the actual traffic light timing
        gradient[i] = (base_value - (base_value - h)) / h; // Simplified numerical gradient
    }
    
    return gradient;
}

} // namespace traffic