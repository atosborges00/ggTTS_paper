function learning_rate = update_learning_rate(epoch, max_epoch)
% Function for updating the learning rate of the neural network during the
% trainig.
%
% Inputs: epoch: integer
%         max_epoch: integer
%
% Output: learning_rate: float

learning_rate = (1)*(1-(epoch/max_epoch));

end