function delta = back_propagation(X_train, hidden_activation, output, output_error, output_weights, learning_rate)
% Function performing the backpropagation algorithm to calculate the
% changes on the neural network weights
%
% Inputs: hidden_activation: (number_hidden_neurons, number_trainig_samples)
%         output: (number_output_neurons, number_trainig_samples)
%         output_weights: (number_output_neurons, number_hidden_neurons)
%         output_error: (number_classes, number_trainig_samples)
%         learning_rate: float
%
% Output: delta.hidden_weights: (number_hidden_neurons, number_attributes)
%         delta.hidden_bias: (number_hidden_neurons, 1)
%         delta.output_weights: (number_output_neurons, number_hidden_neurons)
%         delta.output_bias: (number_output_neurons, 1)

X_train = X_train';

number_trainig_samples = size(output, 2);

% Output layer derivatives
delta_output = output_error .* (output.*(1-output));    % Derivative in relation to the output activation function
delta.output_weights = (1/number_trainig_samples) * learning_rate * delta_output * hidden_activation';   % Derivative in relation to the output weights
delta.output_bias = (1/number_trainig_samples) * learning_rate * sum(delta_output,2); % Derivative in relation to the output bias

% Hidden layer derivatives
delta_hidden = (output_weights' * delta_output) .* (1/2*(1-hidden_activation.^2));  % Derivative in relation to the hidden activation function
delta.hidden_weights = (1/number_trainig_samples) * learning_rate * delta_hidden * X_train'; % Derivative in relation to the hidden weights
delta.hidden_bias = (1/number_trainig_samples) * learning_rate * sum(delta_hidden,2); % Derivative in relation to the hidden bias

end