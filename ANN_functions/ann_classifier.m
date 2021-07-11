function predictions = ann_classifier(X_test, hidden_weights, hidden_bias, output_weights, output_bias)
% Function for making predictions on the test set using the trained
% artificial neural network.
%
% Inputs: X_test: (number_attributes, number_test_samples)
%         hidden_weights: (number_hidden_neurons, number_attributes)
%         hidden_bias: (number_hidden_neurons, 1)
%         output_weights: (number_output_neurons, number_hidden_neurons)
%         output_bias: (number_output_neurons, 1)
%
% Output: predictions: (number_classes, number_test_samples)

output = forward_propagation(X_test, hidden_weights, hidden_bias, output_weights, output_bias);

[~, predictions] = max(output, [], 1);

end