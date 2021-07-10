function [hidden_weights, hidden_bias, output_weights, output_bias] = update_weights(hidden_weights, hidden_bias, output_weights, output_bias, delta)
% Function for initializing the weights of the hidden layer and output
% layer of the neural network by the Nguyen-Widrow method.
%
% Inputs: hidden_weights: (number_hidden_neurons, number_attributes)
%         hidden_bias: (number_hidden_neurons, 1)
%         output_weights: (number_output_neurons, number_hidden_neurons)
%         output_bias: (number_output_neurons, 1)
%         delta: struct [delta.hidden_weights, delta.hidden_bias, delta.output_weights, delta.output_bias]
%
% Output: hidden_weights: (number_hidden_neurons, number_attributes)
%         hidden_bias: (number_hidden_neurons, 1)
%         output_weights: (number_output_neurons, number_hidden_neurons)
%         output_bias: (number_output_neurons, 1)

hidden_weights = hidden_weights + delta.hidden_weights;
hidden_bias = hidden_bias + delta.hidden_bias;
output_weights = output_weights + delta.output_weights;
output_bias = output_bias + delta.output_bias;

end