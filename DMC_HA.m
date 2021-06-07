%% Dados do problema
% Importação do banco de dadados de dertamtologia

clear; close all; clc;
addpath(genpath('phonetic_data'))

% Separação das amostras e os alvos
X = load('FULL_AH_data.txt');
Y = load('AH_targets.txt');
[~,Y] = max(Y,[],2);

%% Classificadores
% Seção que cria as variáveis necessárias e faz a chamada das funções onde
% estão implementados todos os classificadores

% Número de rodadas
Nr = 100;

% Porcentagem de dados usados para treino
train_Percent = 70;

% Definição do número de amostras de treino
[m, p] = size(X);   % tamanho dos dados
n_class = max(Y); % número de classes
num_train = round((train_Percent/100)*m,0); %número de amostras de treino

confusion = zeros(n_class, n_class);   % Matriz de confusão

for rodada = 1:Nr
    
    indTrain = randperm(m, num_train);  % Seleção dos índices de treinamento
    indTest = true(1,m);      %Cria um vetor de "1" lógicos
    indTest(indTrain) = false;  % Torna falso todos os índices que já foram escolhidos
    
    X_trn = X(indTrain,:);  % Separa todos os dados de treino para matriz X
    Y_trn = Y(indTrain,:);  % Separa todos os targets de treino para matriz y
    
    X_tst = X(indTest,:);   % Separa todos os dados de teste para matriz testX
    Y_tst = Y(indTest,:);   % Separa todos os targets de teste para matriz testY
    
    % Declarando uma variável do tipo célula para armazenar as matrizes de
    % covariância de cada classe
    Mcovs = cell(1,6);
    
    % Cálculo das matrizes de ovariância de cada classe
    for i = 1:n_class
        Mcovs{i} = cov(X_trn(Y_trn == i,:));
    end
    
    % Cálculo do centroide de cada classe
    for i = 1:n_class
        centroids(i,:) = mean(X_trn(Y_trn == i,:));
    end
        
    %%%%%%%%%%%%%%%%%%%%%%%%%% CLASSIFICADOR 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Chamada da função que implementa o classificador 3 (Matriz de
    % covariância distinta para cada classe)
    classes = classificador_quadratico(Mcovs, centroids, X_tst);
    
    % Calculando o percentual de acerto do classificador 3
    acertos(rodada) = mean(classes == Y_tst)*100;
    
    % Construção das matrizes de covariância do resultado máximo e mínimo
    for label1 = 1:n_class
        for label2 = 1:n_class
            confusion(label1,label2) = confusion(label1,label2) + sum((classes==label1).*(Y_tst==label2));
        end
    end
    
    for label = 1:n_class
        precision(label) = confusion(label,label)/sum(confusion(:,label));
        recall(label) = confusion(label,label)/sum(confusion(label,:));
    end
    
    f1(rodada) = 2*(mean(precision)*mean(recall))/(mean(precision)+mean(recall));
    
end

%% Resultados
% Seção que reúne e imprime as métricas de avaliação de desempenho de cada
% algoritmo implementado.

%%%%%%%%%%%%%%%%%%%%%%%%%%%% Classificador 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%% 

[maximo, ind_max] = max(acertos);    % Acerto máximo
[minimo, ind_min] = min(acertos);    % Acerto mínimo
media = mean(acertos);    % Acerto médio
dp = std(acertos);        % Desvio padrão

confusion = round(confusion/Nr);

% Exibição dos resultados
fprintf('\n\nAcerto máximo classificador: %.2f\n', maximo);
fprintf('Acerto mínimo classificador: %.2f\n', minimo);
fprintf('Acerto médio classificador: %.2f\n', media);
fprintf('Desvio padrão 100 rodadas classificador: %.2f\n\n', dp);
fprintf('Matriz de confusão máxima do classificador: \n');
disp(confusion);

media_f1 = mean(f1)
dpf1 = std(f1)