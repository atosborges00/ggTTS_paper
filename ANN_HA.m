%% Importa��o dos dados
clear; clc; close all;

% X  = Vetor de entrada 
dataX = load('FULL_AH_data.txt');
dataX = dataX';                  % dataX deve ser: atibutos(linhas) x amostras (colunas)

% Tamanho do conjunto de dados
[m, p] = size(dataX');           % m = numero de amostras, p = numero de atributos

% y  = Classifica��o das flores presente nas colunas de 5 a 7
datay = load('AH_targets.txt');
datay = datay';
[~, labels] = max(datay,[],1);


%% Defini��o otimizada da arquitetura da rede
% Para otimizar a constru��o de uma RNA, um dos par�metros com metodologia
% estabelecida � a defini��o da arquitetura da rede. Segundo a regra de
% Kolmogorov, o n�mero �timo de neur�nios a ser utilizado na camada oculta
% corresponde ao dobro da camada de entrada somado � 1

% Dimens�o de entradas da rede
input_dim = size(dataX,1);

% Dimens�o da camada oculta (regra de Kolmogorov)
hidden_dim = 2*input_dim + 1;

% Dimens�o da camada de sa�da
output_dim = size(datay,1);


%% Normalizando os dados
% Como as vari�veis de entrada apresentam amplitudes distintas, foi
% necess�rio realizar a normaliza��o dos dados. Os dados foram normalizados
% pela m�dia, onde todos os atributos normalizados e passam a apresentar
% m�dia em torno de 0 e vari�ncia unit�ria. 
% 
% $$x_{norm} = \frac{x - \mu_x}{\sigma_x}$$
%
% Esse tipo de normaliza��o,
% apesar de facilitar o processo de treinamento da rede, n�o promove
% distor��es nos dados.

% Normaliza��o pela m�dia
dataX = (dataX-mean(dataX,2))./ std(dataX,[],2);


%% Inicializando os par�metros de forma otimizada
% Para incializar os par�metros da rede de forma eficiente, foi utilizada a
% metodologia de Nguyen-Widrow, o qual estabelece um padr�o para que a
% inicializa��o dos pesos da camada oculta seja mais proveitosa e caia na
% regi�o onde o neur�nio tem a m�xima otimiza��o atrav�s do backpropagation.
% Os pesos s�o reiniciados a cada rodada de testes e o treinamento �
% novamente iniciado.

% Fator beta da inicializa��o de Nguyen-Widrow:
beta = 0.7*(nthroot(hidden_dim, input_dim));

% N�mero da itera��o
iter=1;

% Vari�vel de checagem do conjunto de valida��o e mse
val_check = 1;
mse = inf;
val_mse = 0;

% ep = N�mero m�ximo de �pocas
ep = 1000;
iter=1; % N�mero da itera��o

% N�mero m�ximo de rodadas de teste
max_rounds = 20;

% Matriz de confus�o
confusion = zeros(output_dim,output_dim);

%% Loop de treinamento otimizado
% Para verificar a verdadeira capacidade da rede em solucionar o problema
% de classifica��o � necess�rio realizar testes que garantam tanto precis�o
% como capacidade de generaliza��o.Os percentuais escolhidos aqui foram de 65% para treino,
% 15% para valida��o e 20% dos dados para teste.
% Al�m disso, a taxa de aprendizado n�o � fixa, passando a
% variar de forma linear conforme as �pocas avan�am. Quanto mais avan�ada a
% �poca de treinamento, menor � a taxa de aprendizado, contribuindo para
% uma converg�ncia mais suave dos pesos.

for rodada = 1:max_rounds
    
    % Inicializa��o dos pesos da camada oculta:
    W1 = (-0.5-0.5).*rand(hidden_dim, input_dim) + 0.5;
    norma_W1 = sqrt(sum((W1.^2),2));
    W1 = (beta*W1)./norma_W1;
    
    % Incializa��o do bias da camada oculta
    b1 = (-beta-beta).*rand(hidden_dim, 1) + beta;
    
    % Inicializa��o dos pesos da camada de sa�da:
    W2 = (-0.5-0.5).*rand(output_dim, hidden_dim) + 0.5;
    norma_W2 = sqrt(sum((W2.^2),2));
    W2 = (beta*W2)./norma_W2;
    
    % Incializa��o do bias da camada oculta
    b2 = (-beta-beta).*rand(output_dim, 1) + beta;

    % Porcentagem de treino
    porc_treino = 65;
    
    % Porcentagem de valida��o
    porc_val = 15;
    
    % Calculando o n�mero de amostras de cada procentagem do conjunto de dados
    n_treino = round((porc_treino/100)*m,0);
    n_val = round((porc_val/100)*m,0);
    n_teste = m - (n_treino + n_val);   % amotras de treino s�o as restantes
    
    % Embaralhando os �ndices
    rand_ind = randperm(m);
    
    % Conjunto de dados de treino
    X = dataX(:,rand_ind(1:n_treino));
    yd = datay(:,rand_ind(1:n_treino));
    
    % Conjunto de dados de valida��o
    valX = dataX(:,rand_ind((n_treino+1):(n_treino+n_val)));
    valy = datay(:,rand_ind((n_treino+1):(n_treino+n_val)));
    
    % Conjunto de dados de treino
    testX = dataX(:,rand_ind((n_treino+n_val+1):end));
    testy = datay(:,rand_ind((n_treino+n_val+1):end));
    
    tic
    
    while iter <= ep && mse >= 1e-6  %&& val_check > -1
        
        % Taxa de aprendizado adaptativa: decaimento exponencial
        alfa = (1)*(1-(iter/ep));
        
        %%%%%%%%%%%%%%%%%%%%%% Forward Propagation %%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Camada oculta
        u1 = (W1 * X) + b1;    % somat�rio de pesos ponderados
        v1 = (1-exp(-2*u1))./(1+exp(-2*u1)); % fun��o de ativa��o tanh
        
        % Camada de sa�da
        u2 = (W2 * v1) + b2;    % somat�rio de pesos ponderados
        v2 = 1.0 ./ (1.0 + exp(-u2));   % fun��o de ativa��o sigm
        
        y = v2; % Sa�da da rede
        
        % Avalia��o do erro da sa�da
        erro = yd-y;
        mse = (1/(2*length(X)))*sum(sum(erro.^2));
        
        %%%%%%%%%%%%%%%%%%%%%% Teste de valida��o %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Camada oculta
        u1_val = (W1 * valX) + b1;    % somat�rio de pesos ponderados
        v1_val = (1-exp(-2*u1_val))./(1+exp(-2*u1_val)); % fun��o de ativa��o tanh
        
        % Camada de sa�da
        u2_val = (W2 * v1_val) + b2;    % somat�rio de pesos ponderados
        v2_val = 1.0 ./ (1.0 + exp(-u2_val));   % fun��o de ativa��o sigm
        
        v_y = v2_val; % Sa�da da rede
        
        % Avalia��o do erro da sa�da
        val_hist_mse = val_mse;
        val_erro = valy - v_y;
        val_mse = (1/(2*length(valX)))*sum(sum(val_erro.^2));
        val_check = val_mse - val_hist_mse;
        
        
        %%%%%%%%%%%%%%%%%%%%%%% Back Propagation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Atualiza��es nos pesos da camada de sa�da
        dv2 = erro.*(v2.*(1-v2));    % Derivada do erro em rela��o a v2
        dW2 = (1/length(X))*alfa* dv2 * v1';   % Derivada do erro em rela��o aos pesos 2
        db2 = (1/length(X))*alfa*sum(dv2,2); % Derivada do erro em real�ao aos termos independentes 2
        
        % Atualiza��es dos pesos da camada oculta
        dv1 = (W2'*dv2).*(1/2*(1-v1.^2)); % Derivada do erro em rela��o a v1
        dW1 = (1/length(X))*alfa * dv1 * X'; % Derivada do erro em rela��o aos pesos 1
        db1 = (1/length(X))*alfa*sum(dv1,2); % Derivada do erro em real�ao aos termos independentes 1

        %%%%%%%%%%%%%%%%%%%%% Atualiza��o dos pesos %%%%%%%%%%%%%%%%%%%%%%
        
        W1 = W1 + dW1;  % Atualiza��o dos pesos da camada oculta
        b1 = b1 + db1;  % Atualiza��o dos bias da camada oculta
        W2 = W2 + dW2;  % Atualiza��o dos pesos da camada de sa�da
        b2 = b2 + db2;  % Atualiza��o dos bias da camada de sa�da
        
        %Vari�vel de contagem de �pocas
        iter = iter+1;
        
    end
    
    iter = 1;
    tempo(rodada) = toc;
    
    % A acur�cia da rede � verificada a cada rodada de testes e armazendada
    % para verifica��o posterior da m�dia de acertos
    
    %%%%%%%%%%%%%%%%%%%%% Verifica��o de acur�cia %%%%%%%%%%%%%%%%%%%%%%
    
    % Camada oculta
    u1 = (W1 * [testX valX]) + b1;    % somat�rio de pesos ponderados
    v1 = (1-exp(-2*u1))./(1+exp(-2*u1)); % fun��o de ativa��o tanh
    
    % Camada de sa�da
    u2 = (W2 * v1) + b2;    % somat�rio de pesos ponderados
    v2 = 1.0 ./ (1.0 + exp(-u2));   % fun��o de ativa��o sigm
    
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

%% Verifica��o de acur�cia da rede otimizada
% Agora com os par�metros otimizados, s�o colhidos os mesmo �ndices de
% desempenho para compara��o.

% Taxa m�dia de acerto
media = mean(acertos);
fprintf('\nResultados da rede otimizada: \n\n');
fprintf('Taxa m�dia de acerto: %.2f%%\n', media);

% Taxa m�xima de acerto
maxima = max(acertos);
fprintf('Taxa m�xima de acerto: %.2f%%\n', maxima);

% Taxa m�nima de acerto
minima = min(acertos);
fprintf('Taxa m�nima de acerto: %.2f%%\n', minima);

% Desvio padr�o
dp = std(acertos);
fprintf('Desvio padr�o: %.3f\n', dp);

% Tempo m�dio de treinamento
tempo_medio = mean(tempo);
fprintf('Tempo m�dio de treinamento: %.3f segundos\n', tempo_medio);

% Matriz de confus�o m�dia
confusion = round(confusion/max_rounds);
fprintf('Matriz de confus�o m�dia: \n');
disp(confusion);

media_f1 = mean(f1)
dpf1 = std(f1)