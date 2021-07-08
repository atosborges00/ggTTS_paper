function [hidden_weights, hidden_bias, output_weights, output_bias] = initialize_weights(input_dim, hidden_dim, output_dim)
% Function for initializing the weights of the hidden layer and output
% layer of the neural network by the Nguyen-Widrow method.
%
% Inputs: input_dim: integer
%         hidden_dim: integer
%         output_dim: integer
%
% Output: hidden_weights: (number_hidden_neurons, number_attributes)
%         hidden_bias: (number_hidden_neurons, 1)
%         output_weights: (number_output_neurons, number_hidden_neurons)
%         output_bias: (number_output_neurons, 1)

% Nguyen-Widrow initialization factor:
beta = 0.7*(nthroot(hidden_dim, input_dim));

% Hidden layer initialization:
hidden = (-0.5-0.5) .* rand(hidden_dim, input_dim) + 0.5;
norm_hidden_weights = sqrt(sum((hidden.^2), 2));

hidden_weights = (beta * hidden)./norm_hidden_weights;
hidden_bias = (-beta-beta).*rand(hidden_dim, 1) + beta;

% Ouput layer initialization:
output = (-0.5-0.5).*rand(output_dim, hidden_dim) + 0.5;
norm_output_weights = sqrt(sum((output.^2),2));

output_weights = (beta*output)./norm_output_weights;
output_bias = (-beta-beta).*rand(output_dim, 1) + beta;

end