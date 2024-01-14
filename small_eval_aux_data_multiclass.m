function [A_t,W_t,sigma_all] = small_eval_aux_data_multiclass(W_G,X_aux_all,Y_aux_all,epochs,r,C,net1)%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here
flag_plot=0;

N = length(Y_aux_all);
A_t=[];
[c1,c2] = size(W_G{1,5});
W_t=zeros(c1,c2,C);
%for tt = 1:epochs-1
sigma_all=[];
for c=0:C-1
    ll = find(Y_aux_all==num2str(c));
    Y_aux_c = Y_aux_all(ll);
    X_aux_c = X_aux_all(:,:,:,ll);
    [W_c,netc] = small_NN_train_local_MNIST_multiclass(X_aux_c,Y_aux_c,epochs,N,W_G,flag_plot,C);
    A_t = [A_t,(W_G{1,6}-W_c{1,6})];
    W_t(:,:,c+1) = W_c{1,5};
    W_c = W_G;
    
    sigma_C=[];
    sigma_c = predict(net1,X_aux_c);
    sigma_C=[sigma_C,mean(sigma_c',2)];
    
    for ii=1:epochs
        [W_c,netc] = small_NN_train_local_MNIST_multiclass(X_aux_c,Y_aux_c,1,N,W_c,flag_plot,C);%only 1 epoch
        sigma_c = predict(netc,X_aux_c);
        if(ii<epochs)
        sigma_C=[sigma_C,mean(sigma_c',2)];
        end
        %sigma_C=[sigma_C,mean(sigma_c',2)];
        
    end
    
    sigma_all = [sigma_all, sum(sigma_C,2)];
    
end



end

