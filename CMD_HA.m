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

% Seting data dimensions
[number_samples, number_features] = size(X);

% Number of classes
number_classes = max(Y);

% Trainig dataset setup
train_Percent = 70;

% Confusion matriz setup
confusion = zeros(number_classes, number_classes);

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
        
    %%%%%%%%%%%%%%%%%%%%%%%%%% CLASSIFICADOR 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Chamada da função que implementa o classificador 3 (Matriz de
    % covariância distinta para cada classe)
    classes = quadratic_classifier(covariance_matrices, centroids, X_test);
    
    % Calculando o percentual de acerto do classificador 3
    acertos(test_round) = mean(classes == Y_test)*100;
    
    % Construção das matrizes de covariância do resultado máximo e mínimo
    for label1 = 1:number_classes
        for label2 = 1:number_classes
            confusion(label1,label2) = confusion(label1,label2) + sum((classes==label1).*(Y_test==label2));
        end
    end
    
    for label = 1:number_classes
        precision(label) = confusion(label,label)/sum(confusion(:,label));
        recall(label) = confusion(label,label)/sum(confusion(label,:));
    end
    
    f1(test_round) = 2*(mean(precision)*mean(recall))/(mean(precision)+mean(recall));
    
end

%% Resultados
% Seção que reúne e imprime as métricas de avaliação de desempenho de cada
% algoritmo implementado.

%%%%%%%%%%%%%%%%%%%%%%%%%%%% Classificador 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%% 

[maximo, ind_max] = max(acertos);    % Acerto máximo
[minimo, ind_min] = min(acertos);    % Acerto mínimo
media = mean(acertos);    % Acerto médio
dp = std(acertos);        % Desvio padrão

confusion = round(confusion/max_test_rounds);

% Exibição dos resultados
fprintf('\n\nAcerto máximo classificador: %.2f\n', maximo);
fprintf('Acerto mínimo classificador: %.2f\n', minimo);
fprintf('Acerto médio classificador: %.2f\n', media);
fprintf('Desvio padrão 100 rodadas classificador: %.2f\n\n', dp);
fprintf('Matriz de confusão máxima do classificador: \n');
disp(confusion);

media_f1 = mean(f1)
dpf1 = std(f1)