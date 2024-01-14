function [p_est] = small_soft_label_est_multiclass(net,X_aux_all,Y_aux_all,W_G,W_local,epochs,B,N,r,C,sigma_all)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
sigma_all = calculate_softlables_multiclass(net,X_aux_all,Y_aux_all,C);
b_T = W_local;
b_0 = W_G;

delta_b = (b_T-b_0)./(epochs*r)+ (1./epochs)*(exp(b_0)./sum(exp(b_0)));

const = (epochs-1)./epochs;
AA = eye(C)-(const*sigma_all);
%AA = [AA;ones(1,C)];
%p_est  = lsqnonneg(double(AA),double([delta_b;1]));
p_est = AA\delta_b;
end









