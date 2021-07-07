function predictions = kNN_classifier(X_test, Y_train, nearest_neighbours)
% Function implementing the prediction of the kNN classifier, where the
% class is set to the most common class among the nearest neighbours
%
% Inputs: X_test: (number_test_samples, number_attributes)
%         Y_train: (number_train_samples, 1)
%         nearest_neighbours: (number_test_samples, k_neighbours)
%
% Output: predictions: (number_test_samples, 1)

number_samples = length(X_test);
predictions = zeros(number_samples, 1);

for sample = 1:number_samples
    predictions(sample) = mode( Y_train(nearest_neighbours(sample,:)) );
end

end