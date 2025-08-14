#pragma once

#include "traffic_types.hpp"
#include "traffic_network.hpp"
#include <vector>
#include <functional>

namespace traffic {

class TrafficOptimizer {
public:
    TrafficOptimizer(std::shared_ptr<TrafficNetwork> network);
    ~TrafficOptimizer() = default;

    // Signal timing optimization
    OptimizationResult optimize_signal_timings(OptimizationObjective objective,
                                              const std::vector<std::string>& intersection_ids = {});
    
    // Flow optimization
    OptimizationResult optimize_flow_distribution(OptimizationObjective objective);
    
    // Route optimization
    std::vector<Route> optimize_routes_for_demand(
        const std::vector<std::pair<std::string, std::string>>& origin_destination_pairs);
    
    // Dynamic lane allocation
    std::unordered_map<std::string, int> optimize_lane_allocation(
        const std::vector<std::string>& segment_ids);
    
    // Incident response
    OptimizationResult optimize_for_incident(const std::string& incident_segment_id,
                                            double capacity_reduction_factor);
    
    // Predictive optimization
    OptimizationResult optimize_for_predicted_demand(
        const std::unordered_map<std::string, double>& predicted_flows,
        int time_horizon_minutes = 30);

    // Utility functions
    double evaluate_objective(OptimizationObjective objective) const;
    std::vector<double> get_optimization_history() const { return optimization_history_; }
    
    // Configuration
    void set_optimization_parameters(double tolerance = 1e-6, int max_iterations = 1000);
    void set_weights(double travel_time_weight = 1.0, double congestion_weight = 1.0);

private:
    std::shared_ptr<TrafficNetwork> network_;
    double tolerance_;
    int max_iterations_;
    double travel_time_weight_;
    double congestion_weight_;
    std::vector<double> optimization_history_;

    // Optimization algorithms
    OptimizationResult gradient_descent_optimization(OptimizationObjective objective,
                                                   const std::vector<std::string>& variables);
    OptimizationResult genetic_algorithm_optimization(OptimizationObjective objective);
    OptimizationResult simulated_annealing_optimization(OptimizationObjective objective);
    
    // Objective functions
    double calculate_total_travel_time() const;
    double calculate_total_congestion() const;
    double calculate_total_throughput() const;
    double calculate_flow_balance() const;
    
    // Constraint handling
    bool satisfies_constraints(const OptimizationResult& result) const;
    OptimizationResult apply_constraints(const OptimizationResult& result) const;
    
    // Utility methods
    std::vector<double> compute_gradient(OptimizationObjective objective,
                                       const std::vector<std::string>& variables) const;
    double line_search(const std::vector<double>& direction,
                      const std::vector<std::string>& variables,
                      OptimizationObjective objective) const;
};

// Mathematical optimization utilities
namespace optimization {
    
    // Linear algebra operations for traffic flow matrices
    std::vector<std::vector<double>> matrix_multiply(
        const std::vector<std::vector<double>>& A,
        const std::vector<std::vector<double>>& B);
    
    std::vector<double> solve_linear_system(
        const std::vector<std::vector<double>>& A,
        const std::vector<double>& b);
    
    // Calculus-based optimization
    double partial_derivative(std::function<double(const std::vector<double>&)> func,
                            const std::vector<double>& x, int variable_index,
                            double h = 1e-8);
    
    std::vector<double> gradient(std::function<double(const std::vector<double>&)> func,
                               const std::vector<double>& x);
    
    // Graph theory algorithms
    std::vector<std::vector<double>> floyd_warshall(const std::vector<std::vector<double>>& graph);
    std::vector<int> topological_sort(const std::vector<std::vector<int>>& adjacency_list);
    
    // Network flow algorithms
    double max_flow(const std::vector<std::vector<double>>& capacity_matrix,
                   int source, int sink);
    
    std::pair<double, std::vector<std::vector<double>>> min_cost_flow(
        const std::vector<std::vector<double>>& capacity_matrix,
        const std::vector<std::vector<double>>& cost_matrix,
        const std::vector<double>& supply_demand);

} // namespace optimization

} // namespace traffic