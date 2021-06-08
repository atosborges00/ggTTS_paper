function inv_covariance_matrices = invert_covariance_matrices(covariance_matrices)
% Function responsible for inverting the covariance matrices given in the
% input.
%
% Inputs: covariance_matrices: cell(1, number_classes)
% Outputs: inv_covariance_matrices: cell(1, number_classes)

number_classes = size(covariance_matrices, 2);

inv_covariance_matrices = cell(1,number_classes);

for i = 1:number_classes
    inv_covariance_matrices{i} = pinv(covariance_matrices{i});
end

end