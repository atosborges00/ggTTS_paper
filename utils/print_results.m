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

fprintf('\nMaximum accuracy: %.2f\n', maximum_accuracy);
fprintf('Minimum accuracy: %.2f\n', minimum_accuracy);
fprintf('Mean accuracy: %.2f\n', mean_accuracy);
fprintf('Accuracy standard deviation: %.2f\n', accuracy_std);

fprintf('\nMean F1 score: %.5f\n', mean_f1_score);
fprintf('F1 score standard deviation: %.5f\n\n', f1_score_std);

end