function classes = quadratic_classifier(covariance_matrices, centroids, X_test)
% Function performing the classification. It is the implementation of
% a quadratic distance classifier with the centroids as reference.
%
% Inputs: covariance_matrices: (1, number_classes)
%         centroids: (number_classes, number_features)
%         X_test: (number_test_samples, number_features)
%
% Output: classes: (number_test_samples, 1)

number_classes = size(centroids,1);
[number_samples, ~] = size(X_test);

% Initiating the output variable
classes = zeros(number_samples, 1);

% Initiating the inverse covariance matrices variable
inv_covariance_matrices = cell(1,number_classes);

for i = 1:number_classes
    inv_covariance_matrices{i} = pinv(covariance_matrices{i});
end

for i = 1:number_samples
    distances = zeros(1,number_classes);
    
    for j = 1:number_classes
        distances(1,j) = (X_test(i,:)-centroids(j,:)) * inv_covariance_matrices{j} * (X_test(i,:)-centroids(j,:))';
    end
    
    [~, d_idx] = min(distances);
    classes(i,1) = d_idx;
end

end