function covariance_matrices = get_covariance_matrices(X_train, Y_train)
% Function responsible for calculating the covariance matrix for each
% class in the dataset to be used in the distance calculation.
%
% Inputs: X_train: (number_samples, number_features)
%         Y_train: (number_samples, 1)
%
% Outputs: covariance_matrices: cell(1, number_classes)

number_classes = max(Y_train);

covariance_matrices = cell(1,number_classes);
    
for class = 1:number_classes
    covariance_matrices{class} = cov(X_train(Y_train == class,:));
end

end