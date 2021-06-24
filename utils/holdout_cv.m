function [X_train, Y_train, X_test, Y_test] = holdout_cv(X, Y, train_Percent)
% Function performing the holdout cross validation to separate the trainig
% samples from the test samples.
%
% Inputs: X: (number_samples, number_features)
%         Y: (number_samples, 1)
%         train_Percent: float
%
% Output: X_train: (number_trainig_samples, number_features)
%         Y_train: (number_trainig_samples, 1)
%         X_test: (number_test_samples, number_features)
%         Y_test: (number_test_samples, 1)

number_samples = size(X, 1);

number_training_samples = round((train_Percent/100)*number_samples, 0);

indTrain = randperm(number_samples, number_training_samples);
indTest = true(1,number_samples);
indTest(indTrain) = false;

X_train = X(indTrain,:);
Y_train = Y(indTrain,:);

X_test = X(indTest,:);
Y_test = Y(indTest,:);

end