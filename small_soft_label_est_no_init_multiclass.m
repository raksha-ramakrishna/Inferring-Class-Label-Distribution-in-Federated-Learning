function [p_est] = small_soft_label_est_no_init_multiclass(net,X_aux_all,Y_aux_all,sigma_all,W_G,W_local,epochs,B,N,r,C)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%sigma_cc = calculate_softlables_multiclass(net,X_aux_all,Y_aux_all,C); %final last one
b_T = W_local;
b_0 = W_G;

delta_b = (b_T-b_0)./(epochs*r);%+ (1./epochs)*(exp(b_0)./sum(exp(b_0)));

const = (1)./(epochs); %first epoch already recorded (assume?-not yet)
AA = eye(C)-(const*sigma_all);
%p_est = AA\delta_b;
AA = [AA;ones(1,C)];
p_est  = lsqnonneg(double(AA),double([delta_b;1]));
end









