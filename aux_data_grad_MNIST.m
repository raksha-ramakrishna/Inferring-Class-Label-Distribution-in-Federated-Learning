function [p_est] = aux_data_grad_MNIST(B_G,B_local,A_t,W_t,W_G,W_local,C)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
del_glob_t = W_local-W_G;
ww = vec(del_glob_t');
[s1,s2] = size(ww);
del_W_all = W_t - (repmat(W_G,[1,1,C]));

del_W_matrix = zeros(s1,C);
for i=1:C
    del_W_matrix(:,i) = vec(transpose(del_W_all(:,:,i)));
end
bb = [ww;B_local-B_G];%;1];
AA = [del_W_matrix;(-A_t)];%;ones(1,C)];   
p_est = (double(AA)\double(bb));
%p_est = p_est./sum(p_est);
%lsqnonneg(double(AA),double(bb));
end