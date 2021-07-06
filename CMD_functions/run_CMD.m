function run_CMD(X, Y, train_percent, max_test_rounds)
% Function for runnig the complete CMD algorithm on a given dataset. Joins
% all functions related to the algorithm in the correct order.
%
% Inputs: X: (number_samples, number_attributes)
%         Y: (number_samples, number_classes)
%         train_percent: float
%         max_test_rounds: integer
%
% Output: Text on console

% Switching one-hot-encoded format for numerical categories
[~,Y] = max(Y,[],2);

% Performing mean normalization on the dataset
X = mean_normalization(X);

% Number of classes
number_classes = max(Y);

%% Centroid Minimum Distance main loop
for test_round = 1:max_test_rounds

    % Holdout cross-validation
    [X_train, Y_train, X_test, Y_test] = holdout_cv(X, Y, train_percent);

    % Calculating the covariance matrices for each class on the dataset
    covariance_matrices = get_covariance_matrices(X_train, Y_train);

    % Cálculo do centroide de cada classe
    for i = 1:number_classes
        centroids(i,:) = mean(X_train(Y_train == i,:));
    end

    % Classification by the quadratic classifier
    classes = quadratic_classifier(covariance_matrices, centroids, X_test);

    % Calculating the accuracy achived
    accuracy(test_round) = mean(classes == Y_test)*100;

    % Calculating confusion matrix
    confusion_matrix = get_confusion_matrix(classes, Y_test);

    % Calculating the F1 score of the model
    f1_score(test_round) = get_f1_score(confusion_matrix);

end

%% Results

% Printing all the final results
print_results(accuracy, f1_score)

end