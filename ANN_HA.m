% Cleaning and adding the subfolders to the MATLAB path
clear; close all; clc;
addpath(genpath('phonetic_data'))
addpath(genpath('ANN_functions'))
addpath(genpath('utils'))

% X  = Vetor de entrada 
dataX = load('HA_data.txt');
dataX = dataX';                  % dataX deve ser: atibutos(linhas) x amostras (colunas)

% Tamanho do conjunto de dados
[m, p] = size(dataX');           % m = numero de amostras, p = numero de atributos

% y  = Classificação das flores presente nas colunas de 5 a 7
datay = load('HA_targets.txt');
datay = datay';
[~, labels] = max(datay,[],1);


%% Definição otimizada da arquitetura da rede
% Para otimizar a construção de uma RNA, um dos parâmetros com metodologia
% estabelecida é a definição da arquitetura da rede. Segundo a regra de
% Kolmogorov, o número ótimo de neurônios a ser utilizado na camada oculta
% corresponde ao dobro da camada de entrada somado à 1

% Dimensão de entradas da rede
input_dim = size(dataX,1);

% Dimensão da camada oculta (regra de Kolmogorov)
hidden_dim = 2*input_dim + 1;

% Dimensão da camada de saída
output_dim = size(datay,1);


%% Normalizando os dados
% Como as variáveis de entrada apresentam amplitudes distintas, foi
% necessário realizar a normalização dos dados. Os dados foram normalizados
% pela média, onde todos os atributos normalizados e passam a apresentar
% média em torno de 0 e variância unitária. 
% 
% $$x_{norm} = \frac{x - \mu_x}{\sigma_x}$$
%
% Esse tipo de normalização,
% apesar de facilitar o processo de treinamento da rede, não promove
% distorções nos dados.

% Normalização pela média
dataX = (dataX-mean(dataX,2))./ std(dataX,[],2);


%% Inicializando os parâmetros de forma otimizada
% Para incializar os parâmetros da rede de forma eficiente, foi utilizada a
% metodologia de Nguyen-Widrow, o qual estabelece um padrão para que a
% inicialização dos pesos da camada oculta seja mais proveitosa e caia na
% região onde o neurônio tem a máxima otimização através do backpropagation.
% Os pesos são reiniciados a cada rodada de testes e o treinamento é
% novamente iniciado.

% Fator beta da inicialização de Nguyen-Widrow:
beta = 0.7*(nthroot(hidden_dim, input_dim));

% Número da iteração
epoch=1;

% Variável de checagem do conjunto de validação e mse
val_check = 1;
mse = inf;
val_mse = 0;

% ep = Número máximo de épocas
max_epoch = 1000;
epoch=1; % Número da iteração

% Número máximo de rodadas de teste
max_rounds = 20;

% Matriz de confusão
confusion = zeros(output_dim,output_dim);

%% Loop de treinamento otimizado
% Para verificar a verdadeira capacidade da rede em solucionar o problema
% de classificação é necessário realizar testes que garantam tanto precisão
% como capacidade de generalização.Os percentuais escolhidos aqui foram de 65% para treino,
% 15% para validação e 20% dos dados para teste.
% Além disso, a taxa de aprendizado não é fixa, passando a
% variar de forma linear conforme as épocas avançam. Quanto mais avançada a
% época de treinamento, menor é a taxa de aprendizado, contribuindo para
% uma convergência mais suave dos pesos.

for rodada = 1:max_rounds
    
    % Initializing neural network weights
    [hidden_weights, hidden_bias, output_weights, output_bias] = initialize_weights(input_dim, hidden_dim, output_dim);

    % Porcentagem de treino
    porc_treino = 65;
    
    % Porcentagem de validação
    porc_val = 15;
    
    % Calculando o número de amostras de cada procentagem do conjunto de dados
    n_treino = round((porc_treino/100)*m,0);
    n_val = round((porc_val/100)*m,0);
    n_teste = m - (n_treino + n_val);   % amotras de treino são as restantes
    
    % Embaralhando os índices
    rand_ind = randperm(m);
    
    % Conjunto de dados de treino
    X = dataX(:,rand_ind(1:n_treino));
    yd = datay(:,rand_ind(1:n_treino));
    
    % Conjunto de dados de validação
    valX = dataX(:,rand_ind((n_treino+1):(n_treino+n_val)));
    valy = datay(:,rand_ind((n_treino+1):(n_treino+n_val)));
    
    % Conjunto de dados de treino
    testX = dataX(:,rand_ind((n_treino+n_val+1):end));
    testy = datay(:,rand_ind((n_treino+n_val+1):end));
    
    tic
    
    while epoch <= max_epoch && mse >= 1e-6  %&& val_check > -1
        
        % Taxa de aprendizado adaptativa: decaimento exponencial
        learning_rate = (1)*(1-(epoch/max_epoch));
        
        %%%%%%%%%%%%%%%%%%%%%% Forward Propagation %%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Camada oculta
        u1 = (hidden_weights * X) + hidden_bias;    % somatório de pesos ponderados
        v1 = (1-exp(-2*u1))./(1+exp(-2*u1)); % função de ativação tanh
        
        % Camada de saída
        u2 = (output_weights * v1) + output_bias;    % somatório de pesos ponderados
        v2 = 1.0 ./ (1.0 + exp(-u2));   % função de ativação sigm
        
        y = v2; % Saída da rede
        
        % Avaliação do erro da saída
        erro = yd-y;
        mse = (1/(2*length(X)))*sum(sum(erro.^2));
        
        %%%%%%%%%%%%%%%%%%%%%% Teste de validação %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Camada oculta
        u1_val = (hidden_weights * valX) + hidden_bias;    % somatório de pesos ponderados
        v1_val = (1-exp(-2*u1_val))./(1+exp(-2*u1_val)); % função de ativação tanh
        
        % Camada de saída
        u2_val = (output_weights * v1_val) + output_bias;    % somatório de pesos ponderados
        v2_val = 1.0 ./ (1.0 + exp(-u2_val));   % função de ativação sigm
        
        v_y = v2_val; % Saída da rede
        
        % Avaliação do erro da saída
        val_hist_mse = val_mse;
        val_erro = valy - v_y;
        val_mse = (1/(2*length(valX)))*sum(sum(val_erro.^2));
        val_check = val_mse - val_hist_mse;
        
        
        %%%%%%%%%%%%%%%%%%%%%%% Back Propagation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Atualizações nos pesos da camada de saída
        dv2 = erro.*(v2.*(1-v2));    % Derivada do erro em relação a v2
        dW2 = (1/length(X))*learning_rate* dv2 * v1';   % Derivada do erro em relação aos pesos 2
        db2 = (1/length(X))*learning_rate*sum(dv2,2); % Derivada do erro em realçao aos termos independentes 2
        
        % Atualizações dos pesos da camada oculta
        dv1 = (output_weights'*dv2).*(1/2*(1-v1.^2)); % Derivada do erro em relação a v1
        dW1 = (1/length(X))*learning_rate * dv1 * X'; % Derivada do erro em relação aos pesos 1
        db1 = (1/length(X))*learning_rate*sum(dv1,2); % Derivada do erro em realçao aos termos independentes 1

        %%%%%%%%%%%%%%%%%%%%% Atualização dos pesos %%%%%%%%%%%%%%%%%%%%%%
        
        hidden_weights = hidden_weights + dW1;  % Atualização dos pesos da camada oculta
        hidden_bias = hidden_bias + db1;  % Atualização dos bias da camada oculta
        output_weights = output_weights + dW2;  % Atualização dos pesos da camada de saída
        output_bias = output_bias + db2;  % Atualização dos bias da camada de saída
        
        %Variável de contagem de épocas
        epoch = epoch+1;
        
    end
    
    epoch = 1;
    tempo(rodada) = toc;
    
    % A acurácia da rede é verificada a cada rodada de testes e armazendada
    % para verificação posterior da média de acertos
    
    %%%%%%%%%%%%%%%%%%%%% Verificação de acurácia %%%%%%%%%%%%%%%%%%%%%%
    
    % Camada oculta
    u1 = (hidden_weights * [testX valX]) + hidden_bias;    % somatório de pesos ponderados
    v1 = (1-exp(-2*u1))./(1+exp(-2*u1)); % função de ativação tanh
    
    % Camada de saída
    u2 = (output_weights * v1) + output_bias;    % somatório de pesos ponderados
    v2 = 1.0 ./ (1.0 + exp(-u2));   % função de ativação sigm
    
    [~, pred] = max(v2,[],1);
    
    % Indices de yd
    [~, ind_yd] = max([testy valy],[],1);
    
    acertos(rodada) = mean(ind_yd == pred)*100;
    
    for label1 = 1:output_dim
        for label2 = 1:output_dim
            confusion(label1,label2) = confusion(label1,label2) + sum((ind_yd==label1).*(pred==label2));
        end
    end
    
    for label = 1:output_dim
        precision(label) = confusion(label,label)/sum(confusion(:,label));
        recall(label) = confusion(label,label)/sum(confusion(label,:));
    end
    
    f1(rodada) = 2*(mean(precision)*mean(recall))/(mean(precision)+mean(recall));

end

%% Verificação de acurácia da rede otimizada
% Agora com os parâmetros otimizados, são colhidos os mesmo índices de
% desempenho para comparação.

% Taxa média de acerto
media = mean(acertos);
fprintf('\nResultados da rede otimizada: \n\n');
fprintf('Taxa média de acerto: %.2f%%\n', media);

% Taxa máxima de acerto
maxima = max(acertos);
fprintf('Taxa máxima de acerto: %.2f%%\n', maxima);

% Taxa mínima de acerto
minima = min(acertos);
fprintf('Taxa mínima de acerto: %.2f%%\n', minima);

% Desvio padrão
dp = std(acertos);
fprintf('Desvio padrão: %.3f\n', dp);

% Tempo médio de treinamento
tempo_medio = mean(tempo);
fprintf('Tempo médio de treinamento: %.3f segundos\n', tempo_medio);

% Matriz de confusão média
confusion = round(confusion/max_rounds);
fprintf('Matriz de confusão média: \n');
disp(confusion);

media_f1 = mean(f1)
dpf1 = std(f1)