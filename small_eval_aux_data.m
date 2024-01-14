function [A_t] = small_eval_aux_data(W_G,X_aux_0,Y_aux_0,X_aux_1,Y_aux_1,epochs,r,s1,s2)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
flag_plot=0;
eval_all_t0 =[];
eval_all_t1 = [];
N = length(Y_aux_1);
%for tt = 1:epochs-1
    [W_0,net0] = small_NN_train_local_MNIST(X_aux_0,Y_aux_0,epochs,N,W_G,flag_plot,s1,s2);
    sigma_0 = predict(net0,X_aux_0);
    [W_1,net1] = small_NN_train_local_MNIST(X_aux_1,Y_aux_1,epochs,N,W_G,flag_plot,s1,s2);
    sigma_1 = predict(net1,X_aux_1);
    
    eval_all_t0 = [eval_all_t0;(W_G{1,6}-W_0{1,6})];
    eval_all_t1 = [eval_all_t1;(W_G{1,6}-W_1{1,6})];

    A_t  = [eval_all_t0,eval_all_t1];


end





