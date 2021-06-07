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
number_training_samples = round((train_Percent/100)*number_samples,0);

% Confusion matriz setup
confusion = zeros(number_classes, number_classes);

% Number of maximum rounds of test
max_test_rounds = 100;

%% Centroid Minimum Distance algorithm

for test_round = 1:max_test_rounds
    
    indTrain = randperm(number_samples, number_training_samples);  % Sele��o dos �ndices de treinamento
    indTest = true(1,number_samples);      %Cria um vetor de "1" l�gicos
    indTest(indTrain) = false;  % Torna falso todos os �ndices que j� foram escolhidos
    
    X_trn = X(indTrain,:);  % Separa todos os dados de treino para matriz X
    Y_trn = Y(indTrain,:);  % Separa todos os targets de treino para matriz y
    
    X_tst = X(indTest,:);   % Separa todos os dados de teste para matriz testX
    Y_tst = Y(indTest,:);   % Separa todos os targets de teste para matriz testY
    
    % Declarando uma vari�vel do tipo c�lula para armazenar as matrizes de
    % covari�ncia de cada classe
    Mcovs = cell(1,6);
    
    % C�lculo das matrizes de ovari�ncia de cada classe
    for i = 1:number_classes
        Mcovs{i} = cov(X_trn(Y_trn == i,:));
    end
    
    % C�lculo do centroide de cada classe
    for i = 1:number_classes
        centroids(i,:) = mean(X_trn(Y_trn == i,:));
    end
        
    %%%%%%%%%%%%%%%%%%%%%%%%%% CLASSIFICADOR 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Chamada da fun��o que implementa o classificador 3 (Matriz de
    % covari�ncia distinta para cada classe)
    classes = classificador_quadratico(Mcovs, centroids, X_tst);
    
    % Calculando o percentual de acerto do classificador 3
    acertos(test_round) = mean(classes == Y_tst)*100;
    
    % Constru��o das matrizes de covari�ncia do resultado m�ximo e m�nimo
    for label1 = 1:number_classes
        for label2 = 1:number_classes
            confusion(label1,label2) = confusion(label1,label2) + sum((classes==label1).*(Y_tst==label2));
        end
    end
    
    for label = 1:number_classes
        precision(label) = confusion(label,label)/sum(confusion(:,label));
        recall(label) = confusion(label,label)/sum(confusion(label,:));
    end
    
    f1(test_round) = 2*(mean(precision)*mean(recall))/(mean(precision)+mean(recall));
    
end

%% Resultados
% Se��o que re�ne e imprime as m�tricas de avalia��o de desempenho de cada
% algoritmo implementado.

%%%%%%%%%%%%%%%%%%%%%%%%%%%% Classificador 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%% 

[maximo, ind_max] = max(acertos);    % Acerto m�ximo
[minimo, ind_min] = min(acertos);    % Acerto m�nimo
media = mean(acertos);    % Acerto m�dio
dp = std(acertos);        % Desvio padr�o

confusion = test_round(confusion/Nr);

% Exibi��o dos resultados
fprintf('\n\nAcerto m�ximo classificador: %.2f\n', maximo);
fprintf('Acerto m�nimo classificador: %.2f\n', minimo);
fprintf('Acerto m�dio classificador: %.2f\n', media);
fprintf('Desvio padr�o 100 rodadas classificador: %.2f\n\n', dp);
fprintf('Matriz de confus�o m�xima do classificador: \n');
disp(confusion);

media_f1 = mean(f1)
dpf1 = std(f1)