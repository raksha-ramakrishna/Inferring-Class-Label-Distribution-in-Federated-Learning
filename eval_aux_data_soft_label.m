function [A_t] = eval_aux_data_soft_label(W_G,X_aux_0,Y_aux_0,X_aux_1,Y_aux_1,epochs,r)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
flag_plot=0;
eval_all_t0 =[];
eval_all_t1 = [];
N = length(Y_aux_1);

W_0 = W_G; W_1 = W_G;

for tt = 1:epochs-1
    [W_0,net0] = NN_train_local_MNIST(X_aux_0,Y_aux_0,1,N,W_0,flag_plot);
    sigma_0 = predict(net0,X_aux_0);
    [W_1,net1] = NN_train_local_MNIST(X_aux_1,Y_aux_1,1,N,W_1,flag_plot);
    sigma_1 = predict(net1,X_aux_1);
    
    
%     eval_all_t0 = [eval_all_t0,mean(sigma_0',2)];
%     eval_all_t1 = [eval_all_t1,mean(sigma_1',2)];

   tmp0 = [1;0]-(W_G{1,8}-W_0{1,8})./(epochs*r);
   tmp1 = [0;1]-(W_G{1,8}-W_1{1,8})./(epochs*r);
   
   eval_all_t0 = [eval_all_t0,tmp0];
   eval_all_t1 = [eval_all_t1,tmp1];


end
  %A_t  = [mean(eval_all_t0,2),mean(eval_all_t1,2)];
  A_t = [mean(eval_all_t0,2),mean(eval_all_t1,2)];
end

