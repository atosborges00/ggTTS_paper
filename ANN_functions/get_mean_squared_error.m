function [output_error, mse] = get_mean_squared_error(Y_train, output)
% Function for calculating the Mean Squared Error on the output of the
% neural network.
%
% Inputs: Y_train: (number_classes, number_trainig_samples)
%         output: (number_classes, number_trainig_samples)
%
% Output: output_error: (number_classes, number_trainig_samples)
%         mse: float

number_trainig_samples = size(Y_train, 2);

output_error = Y_train - output;
mse = (1/(2*number_trainig_samples)) * sum(sum(output_error.^2));

end