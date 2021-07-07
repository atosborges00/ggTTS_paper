function f1_score = get_f1_score(confusion_matrix)
% Function responsible for calculating the F1 score of the model's
% predictions.
%
% Inputs: confusion_matrix: (number_classes, number_classes)
%
% Output: f1_score: float

number_classes = size(confusion_matrix, 1);

precision = zeros(1, number_classes);
recall = zeros(1, number_classes);

for label = 1:number_classes
    precision(label) = confusion_matrix(label,label)/sum(confusion_matrix(:,label));
    
    if isnan(precision(label))
        precision(label) = 0;
    end
    
    recall(label) = confusion_matrix(label,label)/sum(confusion_matrix(label,:));
    
    if isnan(recall(label))
        recall(label) = 0;
    end
end

f1_score = 2*(mean(precision)*mean(recall))/(mean(precision)+mean(recall));

end