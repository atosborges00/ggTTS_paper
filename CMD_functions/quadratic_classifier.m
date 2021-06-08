function classes = quadratic_classifier(covariance_matrices, centroids, X_test)
% Function performing the classification. It is the implementation of
% a quadratic distance classifier with the centroids as reference.
%
% Inputs: covariance_matrices: cell(1, number_classes)
%         centroids: (number_classes, number_features)
%         X_test: (number_test_samples, number_features)
%
% Output: classes: (number_test_samples, 1)

number_samples = size(X_test, 1);
classes = zeros(number_samples, 1);

inv_covariance_matrices = invert_covariance_matrices(covariance_matrices);

for sample = 1:number_samples
    
    distances = get_distances(inv_covariance_matrices, centroids, X_test(sample,:));
    
    [~, closer_class] = min(distances);
    classes(sample, 1) = closer_class;
end

end