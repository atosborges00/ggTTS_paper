function distances = get_distances(inv_covariance_matrices, centroids, sample)
% Function responsible for calculating the Mahalanobis distance between 
% a sample and each centroid present in the dataset.
%
% Inputs: covariance_matrices: cell(1, number_classes)
%         centroids: (number_classes, number_features)
%         sample: (1, number_features)
%
% Output: distances: (1, number_classes)

number_classes = size(centroids, 1);

distances = zeros(1,number_classes);
    
for class = 1:number_classes
    distances(1,class) = (sample - centroids(class,:)) * inv_covariance_matrices{class} * (sample - centroids(class,:))';
end

end