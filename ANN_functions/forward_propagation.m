function [output, hidden_activation] = forward_propagation(X_train, hidden_weights, hidden_bias, output_weights, output_bias)
% Function for implementing the neural network forward propagation.
%
% Inputs: X_train: (number_attributes, number_trainig_samples)
%         hidden_weights: (number_hidden_neurons, number_attributes)
%         hidden_bias: (number_hidden_neurons, 1)
%         output_weights: (number_output_neurons, number_hidden_neurons)
%         output_bias: (number_output_neurons, 1)
%
% Output: network_output: (number_trainig_samples, number_trainig_samples)

% Hidden layer
linear_hidden_activation = (hidden_weights * X_train) + hidden_bias;
hidden_activation = (1-exp(-2*linear_hidden_activation))./(1+exp(-2*linear_hidden_activation)); % tanh activation

% Output layer
linear_output_activation = (output_weights * hidden_activation) + output_bias;
output = 1.0 ./ (1.0 + exp(-linear_output_activation));  % sigmoid activation

end