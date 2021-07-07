% Cleaning and adding the subfolders to the MATLAB path
clear; close all; clc;
addpath(genpath('phonetic_data'))
addpath(genpath('kNN_functions'))
addpath(genpath('utils'))

%% CARREGANDO E SEPARANDO OS DADOS

dataX = load('HA_data.txt');
datay = load('HA_targets.txt');

[m,n] = size(datay);

porcn = 70;
n_tests = round((porcn/100)*m,0);
max_rodada = 100;

Mconfusao = zeros(n,n);

for rodada = 1:max_rodada
    
    indMem = randperm(m, n_tests);
    indTest = true(1,m);
    indTest(indMem) = false;
    
    X = dataX(indMem,:);
    [~,y] = max(datay(indMem,:),[],2);
    
    testX = dataX(indTest,:);
    [~,testy] = max(datay(indTest,:),[],2);
    
    ind_nearest = knnsearch(X,testX,'K',7,'Distance','mahalanobis','NSMethod','exhaustive');
    
    predictions = kNN_classifier(testX, y, ind_nearest);
    predictions = predictions';
    
    acertos(rodada) = mean(testy == predictions')*100;
    
        % Para a matriz confusão.
    for label1 = 1:n
        for label2 = 1:n
            Mconfusao(label1,label2) = Mconfusao(label1,label2) + sum((testy==label1).*(predictions'==label2));
        end
    end
    
    for label = 1:n
        precision(label) = Mconfusao(label,label)/sum(Mconfusao(:,label));
        
        if isnan(precision(label))
            precision(label) = 0;
        end
        
        recall(label) = Mconfusao(label,label)/sum(Mconfusao(label,:));
        
        if isnan(recall(label))
            recall(label) = 0;
        end
        
    end
    
    f1(rodada) = 2*(mean(precision)*mean(recall))/(mean(precision)+mean(recall));
    
end

media = mean(acertos)
dp = std(acertos)
Mconfusao = round(Mconfusao/max_rodada)

media_f1 = mean(f1)
dpf1 = std(f1)