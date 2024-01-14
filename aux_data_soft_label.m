function [p_est] = aux_data_soft_label(W_G,W_local,A_t,epochs,r)
   b_t = W_G{1,8}; %b_0
    
    b_t_plus = W_local{1,8}; %b_T
    
    bb = [b_t_plus-b_t]./(epochs.*r);
    bb = [bb;1];
    AA = eye(length(b_t))-A_t;
    AA = [AA;[1,1]];
    p_est  = lsqnonneg(double(AA),double(bb));
    
end



