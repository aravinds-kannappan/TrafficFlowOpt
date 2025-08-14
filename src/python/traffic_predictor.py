"""
JAX-based Traffic Flow Prediction Module
Implements GPU-accelerated predictive modeling for traffic optimization
"""

import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
from typing import Dict, List, Tuple, Optional, NamedTuple
import numpy as np
from dataclasses import dataclass
import pandas as pd
from functools import partial

# Enable 64-bit precision for better numerical stability
jax.config.update("jax_enable_x64", True)

@dataclass
class PredictionConfig:
    """Configuration for traffic prediction models"""
    sequence_length: int = 60  # minutes of historical data
    prediction_horizon: int = 30  # minutes to predict ahead
    learning_rate: float = 0.001
    batch_size: int = 32
    hidden_units: int = 128
    num_epochs: int = 1000
    early_stopping_patience: int = 50

class TrafficState(NamedTuple):
    """Traffic network state representation"""
    flows: jnp.ndarray  # [num_segments] - current flow rates
    speeds: jnp.ndarray  # [num_segments] - average speeds
    occupancies: jnp.ndarray  # [num_segments] - occupancy rates
    signal_states: jnp.ndarray  # [num_intersections] - traffic light states
    timestamp: float

class NeuralODEParams(NamedTuple):
    """Parameters for Neural ODE traffic flow model"""
    W1: jnp.ndarray
    b1: jnp.ndarray
    W2: jnp.ndarray
    b2: jnp.ndarray
    W_out: jnp.ndarray
    b_out: jnp.ndarray

class TrafficPredictor:
    """GPU-accelerated traffic flow predictor using JAX"""
    
    def __init__(self, config: PredictionConfig):
        self.config = config
        self.key = random.PRNGKey(42)
        self.model_params = None
        self.is_trained = False
        
    def initialize_params(self, input_dim: int, output_dim: int) -> NeuralODEParams:
        """Initialize neural network parameters"""
        key1, key2, key3, key4, key5, key6 = random.split(self.key, 6)
        
        # Xavier initialization
        scale = jnp.sqrt(2.0 / (input_dim + self.config.hidden_units))
        
        return NeuralODEParams(
            W1=random.normal(key1, (input_dim, self.config.hidden_units)) * scale,
            b1=jnp.zeros(self.config.hidden_units),
            W2=random.normal(key2, (self.config.hidden_units, self.config.hidden_units)) * scale,
            b2=jnp.zeros(self.config.hidden_units),
            W_out=random.normal(key3, (self.config.hidden_units, output_dim)) * scale,
            b_out=jnp.zeros(output_dim)
        )
    
    @partial(jit, static_argnums=(0,))
    def neural_ode_dynamics(self, params: NeuralODEParams, state: jnp.ndarray, t: float) -> jnp.ndarray:
        """Neural ODE dynamics function for traffic flow evolution"""
        # Add time as feature
        augmented_state = jnp.concatenate([state, jnp.array([t])])
        
        # Forward pass through neural network
        h1 = jnp.tanh(jnp.dot(augmented_state, params.W1) + params.b1)
        h2 = jnp.tanh(jnp.dot(h1, params.W2) + params.b2)
        dynamics = jnp.dot(h2, params.W_out) + params.b_out
        
        return dynamics
    
    @partial(jit, static_argnums=(0,))
    def runge_kutta_4(self, params: NeuralODEParams, state: jnp.ndarray, 
                      t: float, dt: float) -> jnp.ndarray:
        """4th order Runge-Kutta integration"""
        k1 = self.neural_ode_dynamics(params, state, t)
        k2 = self.neural_ode_dynamics(params, state + dt * k1 / 2, t + dt / 2)
        k3 = self.neural_ode_dynamics(params, state + dt * k2 / 2, t + dt / 2)
        k4 = self.neural_ode_dynamics(params, state + dt * k3, t + dt)
        
        return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    
    @partial(jit, static_argnums=(0,))
    def predict_trajectory(self, params: NeuralODEParams, initial_state: jnp.ndarray,
                          time_horizon: int) -> jnp.ndarray:
        """Predict traffic flow trajectory using Neural ODE"""
        dt = 1.0  # 1 minute time step
        times = jnp.arange(0, time_horizon, dt)
        
        def ode_step(state, t):
            next_state = self.runge_kutta_4(params, state, t, dt)
            return next_state, next_state
        
        _, trajectory = jax.lax.scan(ode_step, initial_state, times)
        return trajectory
    
    @partial(jit, static_argnums=(0,))
    def traffic_flow_pde(self, flows: jnp.ndarray, speeds: jnp.ndarray,
                        densities: jnp.ndarray, dx: float, dt: float) -> jnp.ndarray:
        """Traffic flow PDE (conservation equation) solver"""
        # ∂ρ/∂t + ∂(ρv)/∂x = 0
        # Using upwind finite difference scheme
        
        flux = flows * speeds
        
        # Spatial derivatives using upwind scheme
        flux_grad = jnp.zeros_like(flux)
        flux_grad = flux_grad.at[1:].set((flux[1:] - flux[:-1]) / dx)
        flux_grad = flux_grad.at[0].set((flux[1] - flux[0]) / dx)  # Forward difference at boundary
        
        # Time evolution
        new_densities = densities - dt * flux_grad
        
        # Apply physical constraints
        new_densities = jnp.maximum(new_densities, 0.0)  # Non-negative density
        new_densities = jnp.minimum(new_densities, 1.0)  # Maximum density
        
        return new_densities
    
    @partial(jit, static_argnums=(0,))
    def macroscopic_fundamental_diagram(self, density: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Macroscopic Fundamental Diagram (MFD) for speed-density relationship"""
        # Greenshields model: v = v_free * (1 - ρ/ρ_jam)
        v_free = 60.0  # Free flow speed (km/h)
        rho_jam = 1.0  # Jam density (normalized)
        
        speed = v_free * (1.0 - density / rho_jam)
        speed = jnp.maximum(speed, 0.0)  # Non-negative speed
        
        # Flow = density * speed
        flow = density * speed
        
        return speed, flow
    
    @partial(jit, static_argnums=(0,))
    def loss_function(self, params: NeuralODEParams, batch_states: jnp.ndarray,
                     batch_targets: jnp.ndarray) -> float:
        """Loss function for training"""
        def single_prediction_loss(state, target):
            prediction = self.predict_trajectory(params, state, self.config.prediction_horizon)
            return jnp.mean((prediction[-1] - target) ** 2)
        
        losses = vmap(single_prediction_loss)(batch_states, batch_targets)
        return jnp.mean(losses)
    
    def prepare_training_data(self, traffic_data: pd.DataFrame) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Prepare training data from historical traffic observations"""
        # Convert pandas DataFrame to JAX arrays
        features = ['flow_rate', 'average_speed', 'occupancy']
        
        # Normalize data
        data_normalized = (traffic_data[features] - traffic_data[features].mean()) / traffic_data[features].std()
        data_array = jnp.array(data_normalized.values)
        
        # Create sequences
        X, y = [], []
        for i in range(len(data_array) - self.config.sequence_length - self.config.prediction_horizon):
            X.append(data_array[i:i + self.config.sequence_length])
            y.append(data_array[i + self.config.sequence_length:i + self.config.sequence_length + self.config.prediction_horizon])
        
        return jnp.array(X), jnp.array(y)
    
    def train(self, traffic_data: pd.DataFrame) -> Dict[str, float]:
        """Train the traffic prediction model"""
        print("Preparing training data...")
        X_train, y_train = self.prepare_training_data(traffic_data)
        
        # Initialize parameters
        input_dim = X_train.shape[-1] + 1  # +1 for time feature
        output_dim = y_train.shape[-1]
        self.model_params = self.initialize_params(input_dim, output_dim)
        
        # Training loop
        print("Starting training...")
        train_losses = []
        best_loss = float('inf')
        patience_counter = 0
        
        # Adam optimizer state
        from jax.example_libraries.optimizers import adam
        opt_init, opt_update, get_params = adam(self.config.learning_rate)
        opt_state = opt_init(self.model_params)
        
        # JIT compile gradient function
        grad_fn = jit(grad(self.loss_function))
        
        for epoch in range(self.config.num_epochs):
            # Random batch sampling
            key, subkey = random.split(self.key)
            batch_indices = random.choice(subkey, len(X_train), (self.config.batch_size,), replace=False)
            batch_X = X_train[batch_indices]
            batch_y = y_train[batch_indices]
            
            # Compute gradients and update parameters
            current_params = get_params(opt_state)
            grads = grad_fn(current_params, batch_X[:, -1], batch_y[:, -1])  # Use last state as initial condition
            opt_state = opt_update(epoch, grads, opt_state)
            
            # Compute loss every 10 epochs
            if epoch % 10 == 0:
                current_params = get_params(opt_state)
                loss = self.loss_function(current_params, batch_X[:, -1], batch_y[:, -1])
                train_losses.append(float(loss))
                
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
                
                # Early stopping
                if loss < best_loss:
                    best_loss = loss
                    patience_counter = 0
                    self.model_params = current_params
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        self.is_trained = True
        return {
            'final_loss': float(best_loss),
            'epochs_trained': epoch + 1,
            'train_losses': train_losses
        }
    
    def predict(self, current_state: TrafficState, time_horizon: int = None) -> jnp.ndarray:
        """Predict future traffic states"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if time_horizon is None:
            time_horizon = self.config.prediction_horizon
        
        # Prepare input state
        state_vector = jnp.concatenate([current_state.flows, current_state.speeds, current_state.occupancies])
        
        # Make prediction
        prediction = self.predict_trajectory(self.model_params, state_vector, time_horizon)
        
        return prediction
    
    def predict_with_uncertainty(self, current_state: TrafficState, 
                                num_samples: int = 100) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Predict with uncertainty quantification using Monte Carlo dropout"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Add noise to parameters for uncertainty estimation
        predictions = []
        noise_scale = 0.01
        
        for _ in range(num_samples):
            key, subkey = random.split(self.key)
            noisy_params = NeuralODEParams(
                W1=self.model_params.W1 + random.normal(subkey, self.model_params.W1.shape) * noise_scale,
                b1=self.model_params.b1,
                W2=self.model_params.W2 + random.normal(subkey, self.model_params.W2.shape) * noise_scale,
                b2=self.model_params.b2,
                W_out=self.model_params.W_out + random.normal(subkey, self.model_params.W_out.shape) * noise_scale,
                b_out=self.model_params.b_out
            )
            
            state_vector = jnp.concatenate([current_state.flows, current_state.speeds, current_state.occupancies])
            pred = self.predict_trajectory(noisy_params, state_vector, self.config.prediction_horizon)
            predictions.append(pred)
        
        predictions = jnp.stack(predictions)
        mean_prediction = jnp.mean(predictions, axis=0)
        std_prediction = jnp.std(predictions, axis=0)
        
        return mean_prediction, std_prediction
    
    def optimize_signal_timing(self, network_state: TrafficState,
                              signal_constraints: Dict[str, Tuple[float, float]]) -> jnp.ndarray:
        """Optimize traffic signal timing using gradient-based optimization"""
        
        def objective(signal_timings):
            # Simulate traffic flow with given signal timings
            predicted_flows = self.predict(network_state)
            
            # Objective: minimize total travel time
            travel_times = predicted_flows / (network_state.speeds + 1e-6)  # Avoid division by zero
            return jnp.sum(travel_times)
        
        # Initial signal timings
        num_signals = len(signal_constraints)
        initial_timings = jnp.array([30.0] * num_signals)  # 30 seconds default
        
        # Gradient-based optimization
        grad_fn = grad(objective)
        learning_rate = 0.1
        
        timings = initial_timings
        for _ in range(100):  # Optimization iterations
            gradients = grad_fn(timings)
            timings = timings - learning_rate * gradients
            
            # Apply constraints
            for i, (min_time, max_time) in enumerate(signal_constraints.values()):
                timings = timings.at[i].set(jnp.clip(timings[i], min_time, max_time))
        
        return timings
    
    def get_model_info(self) -> Dict:
        """Get information about the trained model"""
        if not self.is_trained:
            return {"status": "not_trained"}
        
        # Count parameters
        total_params = sum(p.size for p in jax.tree_util.tree_leaves(self.model_params))
        
        return {
            "status": "trained",
            "total_parameters": int(total_params),
            "sequence_length": self.config.sequence_length,
            "prediction_horizon": self.config.prediction_horizon,
            "hidden_units": self.config.hidden_units
        }