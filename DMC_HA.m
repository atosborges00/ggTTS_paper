%% Dados do problema
% Importa��o do banco de dadados de dertamtologia

clear; close all; clc;
addpath(genpath('phonetic_data'))

% Separa��o das amostras e os alvos
X = load('FULL_AH_data.txt');
Y = load('AH_targets.txt');
[~,Y] = max(Y,[],2);

%% Classificadores
% Se��o que cria as vari�veis necess�rias e faz a chamada das fun��es onde
% est�o implementados todos os classificadores

% N�mero de rodadas
Nr = 100;

% Porcentagem de dados usados para treino
train_Percent = 70;

% Defini��o do n�mero de amostras de treino
[m, p] = size(X);   % tamanho dos dados
n_class = max(Y); % n�mero de classes
num_train = round((train_Percent/100)*m,0); %n�mero de amostras de treino

confusion = zeros(n_class, n_class);   % Matriz de confus�o

for rodada = 1:Nr
    
    indTrain = randperm(m, num_train);  % Sele��o dos �ndices de treinamento
    indTest = true(1,m);      %Cria um vetor de "1" l�gicos
    indTest(indTrain) = false;  % Torna falso todos os �ndices que j� foram escolhidos
    
    X_trn = X(indTrain,:);  % Separa todos os dados de treino para matriz X
    Y_trn = Y(indTrain,:);  % Separa todos os targets de treino para matriz y
    
    X_tst = X(indTest,:);   % Separa todos os dados de teste para matriz testX
    Y_tst = Y(indTest,:);   % Separa todos os targets de teste para matriz testY
    
    % Declarando uma vari�vel do tipo c�lula para armazenar as matrizes de
    % covari�ncia de cada classe
    Mcovs = cell(1,6);
    
    % C�lculo das matrizes de ovari�ncia de cada classe
    for i = 1:n_class
        Mcovs{i} = cov(X_trn(Y_trn == i,:));
    end
    
    % C�lculo do centroide de cada classe
    for i = 1:n_class
        centroids(i,:) = mean(X_trn(Y_trn == i,:));
    end
        
    %%%%%%%%%%%%%%%%%%%%%%%%%% CLASSIFICADOR 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Chamada da fun��o que implementa o classificador 3 (Matriz de
    % covari�ncia distinta para cada classe)
    classes = classificador_quadratico(Mcovs, centroids, X_tst);
    
    % Calculando o percentual de acerto do classificador 3
    acertos(rodada) = mean(classes == Y_tst)*100;
    
    % Constru��o das matrizes de covari�ncia do resultado m�ximo e m�nimo
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
% Se��o que re�ne e imprime as m�tricas de avalia��o de desempenho de cada
% algoritmo implementado.

%%%%%%%%%%%%%%%%%%%%%%%%%%%% Classificador 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%% 

[maximo, ind_max] = max(acertos);    % Acerto m�ximo
[minimo, ind_min] = min(acertos);    % Acerto m�nimo
media = mean(acertos);    % Acerto m�dio
dp = std(acertos);        % Desvio padr�o

confusion = round(confusion/Nr);

% Exibi��o dos resultados
fprintf('\n\nAcerto m�ximo classificador: %.2f\n', maximo);
fprintf('Acerto m�nimo classificador: %.2f\n', minimo);
fprintf('Acerto m�dio classificador: %.2f\n', media);
fprintf('Desvio padr�o 100 rodadas classificador: %.2f\n\n', dp);
fprintf('Matriz de confus�o m�xima do classificador: \n');
disp(confusion);

media_f1 = mean(f1)
dpf1 = std(f1)