function run_ANN(X, Y, train_percent, min_mse, max_epoch, max_rounds)
% Function for runnig the complete ANN algorithm on a given dataset. Joins
% all functions related to the algorithm in the correct order.
%
% Inputs: X: (number_samples, number_attributes)
%         Y: (number_samples, number_classes)
%         train_percent: float
%         min_mse: float
%         max_epoch: integer
%         max_rounds: integer
%
% Output: Text on console

%% ANN Setup

% Neural Network dimentions
input_dim = size(dataX,1);
hidden_dim = 2*input_dim + 1;
output_dim = size(datay,1);

% Performing mean normalization on the dataset
X = mean_normalization(X);

% Initial Mean Squared Error
mse = inf;

%% ANN main loop

for test_round = 1:max_rounds
    
    % Initializing neural network weights
    [hidden_weights, hidden_bias, output_weights, output_bias] = initialize_weights(input_dim, hidden_dim, output_dim);

    % Holdout cross-validation
    [X_train, Y_train, X_test, Y_test] = holdout_cv(X, Y, train_percent);
    
    % Set initial epoch for new round
    epoch = 1;
    
%% ANN trainig

    while epoch <= max_epoch && mse >= min_mse
        
        % Updating the learning rate for each epoch
        learning_rate = update_learning_rate(epoch, max_epoch);
        
        % Forward propagation step
        [output, hidden_activation] = forward_propagation(X_train, hidden_weights, hidden_bias, output_weights, output_bias);
        
        % Calculating the Mean Squared Error of the output
        [output_error, mse] = get_mean_squared_error(Y_train, output);
        
        % Calculating the change needed on the weights
        delta = back_propagation(X_train, hidden_activation, output, output_error, output_weights, learning_rate);
        
        % Updating weights
        [hidden_weights, hidden_bias, output_weights, output_bias] = update_weights(hidden_weights, hidden_bias, output_weights, output_bias, delta);
        
        %Variável de contagem de épocas
        epoch = epoch+1;
        
    end
    
%% ANN performance evaluation

    % Testing accuracy
    predictions = ann_classifier(X_test, hidden_weights, hidden_bias, output_weights, output_bias);
    
    % Converting one hot encoded to single line
    [~, Y_test] = max(Y_test, [], 1);
    
    % Calculating the accuracy achived
    acccuracy(test_round) = mean(Y_test == predictions)*100;
    
    % Calculating confusion matrix
    confusion_matrix = get_confusion_matrix(predictions, Y_test);
    
    % Calculating the F1 score of the model
    f1_score(test_round) = get_f1_score(confusion_matrix);

end

%% Show results

% Printing all the final results
print_results(accuracy, f1_score)

end
