function run_kNN(X, Y, number_neighbours, train_percent, max_test_rounds)
% Function for runnig the complete kNN algorithm on a given dataset. Joins
% all functions related to the algorithm in the correct order.
%
% Inputs: X: (number_samples, number_attributes)
%         Y: (number_samples, number_classes)
%         number_neighbours: odd integer
%         train_percent: float
%         max_test_rounds: integer
%
% Output: Text on console

% Switching one-hot-encoded format for numerical categories
[~,Y] = max(Y,[],2);

% Performing mean normalization on the dataset
X = mean_normalization(X);

%% k Nearest Neighbour main loop

for test_round = 1:max_test_rounds
    
    % Holdout cross-validation
    [X_train, Y_train, X_test, Y_test] = holdout_cv(X, Y, train_percent);
    
    % Finding the k nearest neighbours to each test sample
    nearest_neighbours = knnsearch(X_train, X_test, 'K', number_neighbours, 'Distance', 'mahalanobis', 'NSMethod', 'exhaustive');
    
    % Predicting the class of each test sample by it's nearest neighbours
    predictions = kNN_classifier(X_test, Y_train, nearest_neighbours);
    
    % Calculating the accuracy achived
    accuracy(test_round) = mean(predictions == Y_test)*100;
    
    % Calculating confusion matrix
    confusion_matrix = get_confusion_matrix(predictions, Y_test);
    
    % Calculating the F1 score of the model
    f1_score(test_round) = get_f1_score(confusion_matrix);
    
end

%% Results

% Printing all the final results
print_results(accuracy, f1_score)

end
