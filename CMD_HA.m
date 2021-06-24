%% Centroid Minimum Distance Classifier for the HA Process
% Code author: Atos Borges
% Date: 06/06/2021
% This code was developed and will publish by the authors at the paper:
% AUTOMATIC IDENTIFICATION OF SYNTHETICALLY GENARATED INTERLANGUAGE 
% TRANSFER PHENOMENA BETWEEN BRAZILIAN PORTUGUESE (L1) AND ENGLISH AS 
% FOREIGN LANGUAGE (L2).
%
% You SHOULD NOT use, copy, modify or redistribute any of this code before
% the final paper be published. We will let you know when this code will be
% available for public use.

%% Setup

% Cleaning and adding the subfolders to the MATLAB path
clear; close all; clc;
addpath(genpath('phonetic_data'))
addpath(genpath('CMD_functions'))
addpath(genpath('utils'))

% Loading data files
X = load('FULL_AH_data.txt');
Y = load('AH_targets.txt');

% Switching one-hot-encoded format for numerical categories
[~,Y] = max(Y,[],2);

% Number of classes
number_classes = max(Y);

% Trainig dataset setup
train_Percent = 70;

% Number of maximum rounds of test
max_test_rounds = 100;

%% Centroid Minimum Distance algorithm

for test_round = 1:max_test_rounds
    
    % Holdout cross-validation
    [X_train, Y_train, X_test, Y_test] = holdout_cv(X, Y, train_Percent);
    
    % Calculating the covariance matrices for each class on the dataset
    covariance_matrices = get_covariance_matrices(X_train, Y_train);
    
    % Cálculo do centroide de cada classe
    for i = 1:number_classes
        centroids(i,:) = mean(X_train(Y_train == i,:));
    end
    
    % Classification by the quadratic classifier
    classes = quadratic_classifier(covariance_matrices, centroids, X_test);
    
    % Calculating the accuracy achived
    accuracy(test_round) = mean(classes == Y_test)*100;
    
    % Calculating confusion matrix
    confusion_matrix = get_confusion_matrix(classes, Y_test);
    
    % Calculating the F1 score of the model
    f1_score(test_round) = get_f1_score(confusion_matrix);
    
end

%% Results

% Printing all the final results
print_results(accuracy, f1_score)
