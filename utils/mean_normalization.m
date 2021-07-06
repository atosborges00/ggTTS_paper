function normalized_dataset = mean_normalization(dataset)
% Function responsible for implementing the mean normalization on the
% dataset, where all the samples are centered on 0 with variance 1.
%
% Inputs: dataset: (number_samples, number_attributes)
%
% Output: normalized_dataset: (number_samples, number_attributes)

normalized_dataset = (dataset - mean(dataset))./ std(dataset);

end