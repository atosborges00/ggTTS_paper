function centroids = get_centroids(X, Y)
% Function for calculating the centroid of each class in the dataset.
%
% Inputs: X: (number_samples, number_attributes)
%         Y: (number_samples, number_classes)
%
% Output: centroids: (number_classes, number_attributes)

number_classes = max(Y);

for class = 1:number_classes
    centroids(class,:) = mean(X(Y == class,:));
end

end