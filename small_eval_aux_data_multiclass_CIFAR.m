function [A_t,W_t,sigma_all] = small_eval_aux_data_multiclass_CIFAR(W_G,X_aux_all,Y_aux_all,epochs,r,C,net1)%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here
flag_plot=0;

N = length(Y_aux_all);
A_t=[];
[c1,c2] = size(W_G{1,7});
W_t=zeros(c1,c2,C);
%for tt = 1:epochs-1
%C=10; %number of classes
class_names = categories(Y_aux_all);
sigma_all=[];
for c=1:C
    ll = find(Y_aux_all==class_names{c});
    Y_aux_c = Y_aux_all(ll);
    X_aux_c = X_aux_all(:,:,:,ll);
%     [W_c,netc] = small_NN_train_local_CIFAR_multiclass(X_aux_c,Y_aux_c,epochs,N,W_G,flag_plot,C);
%     A_t = [A_t,(W_G{1,8}-W_c{1,8})];
%     W_t(:,:,c) = W_c{1,7};
    W_c = W_G;
    
    sigma_C=[];
    sigma_c = predict(net1,X_aux_c);
    sigma_C=[sigma_C,mean(sigma_c',2)];
    
    for ii=1:epochs
        [W_c,netc] = small_NN_train_local_CIFAR_multiclass(X_aux_c,Y_aux_c,1,N,W_c,flag_plot,C);%only 1 epoch
        sigma_c = predict(netc,X_aux_c);
        if(ii<epochs)
        sigma_C=[sigma_C,mean(sigma_c',2)];
        end
        
    end
    A_t = [A_t,(W_G{1,8}-W_c{1,8})];
    W_t(:,:,c) = W_c{1,7};
    sigma_all = [sigma_all, sum(sigma_C,2)];
    
end



end

