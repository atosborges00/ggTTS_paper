function print_results(accuracy, f1_score)
% Function for printing the model's final results.
%
% Inputs: accuracy: (1, max_test_rounds)
%         f1_score: (1, max_test_rounds)
%
% Output: Text on console

maximum_accuracy = max(accuracy);
minimum_accuracy = min(accuracy);
mean_accuracy = mean(accuracy);
accuracy_std = std(accuracy);
mean_f1_score = mean(f1_score);
f1_score_std = std(f1_score);

fprintf('\n\nMaximum accuracy: %.2f\n', maximum_accuracy);
fprintf('\n\nMinimum accuracy: %.2f\n', minimum_accuracy);
fprintf('\n\nMean accuracy: %.2f\n', mean_accuracy);
fprintf('\n\nAccuracy standard deviation: %.2f\n', accuracy_std);

fprintf('\n\n\n');

fprintf('\n\nMean F1 score: %.2f\n', mean_f1_score);
fprintf('\n\nF1 score standard deviation: %.2f\n', f1_score_std);

end