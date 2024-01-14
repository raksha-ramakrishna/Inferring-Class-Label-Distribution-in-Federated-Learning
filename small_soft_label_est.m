function [p_est] = small_soft_label_est(net,X_aux_0,X_aux_1,W_G,W_local,epochs,B,N,r)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
[sigma_0,sigma_1] = calculate_softlables(net,X_aux_0,X_aux_1);
b_T = W_local{1,6};
b_0 = W_G{1,6};

delta_b = (b_T-b_0)./(epochs*r)+ (1./epochs)*(exp(b_0)./sum(exp(b_0)));
S_0 = mean(sigma_0',2);
S_1 = mean(sigma_1',2);
const = (epochs-1)./epochs;
AA = eye(2)-(const*[S_0,S_1]);
AA = [AA;[1,1]];
p_est  = lsqnonneg(double(AA),double([delta_b;1]));
end






