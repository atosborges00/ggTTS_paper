function confusion_matrix = get_confusion_matrix(Y_predictions, Y_test)
% Function responsible for building the confusion matrix of the model's
% results
%
% Inputs: Y_predictions: (number_test_samples, 1)
%         Y_test: (number_test_samples, 1)
%
% Output: confusion_matrix: (number_classes, number_classes)

number_classes = max(Y_test);

confusion_matrix = zeros(number_classes, number_classes);

for first_label = 1:number_classes
    for second_label = 1:number_classes
        confusion_matrix(first_label,second_label) = confusion_matrix(first_label,second_label) + sum((Y_predictions==first_label).*(Y_test==second_label));
    end
end

end