function [p_est] = small_aux_data_grad_multiclass(W_G,W_local,A_t,C)
   b_t = W_G; %b_0
   % W_t = vec(W_G{1,7});
    b_t_plus = W_local; %b_T
   % W_t_plus  = vec(W_local{1,7});
    bb = [b_t_plus-b_t];%W_t_plus-W_t];
    bb = [bb;1];
    
    AA = [(-A_t);ones(1,C)];
    p_est = lsqnonneg(double(AA),double(bb));
    %p_est = A_t\bb;
 

end



