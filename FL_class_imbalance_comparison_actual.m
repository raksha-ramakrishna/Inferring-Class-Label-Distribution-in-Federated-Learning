function [p_est] =FL_class_imbalance_comparison_actual(W_t,W_G,W_local,C,n_aux,N_k)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[C,S]=size(W_local); 

del_glob_t = W_local-W_G;
del_W_all = W_t - (repmat(W_G,[1,1,C]));
p_est = zeros(C,1);

for p = 1:C
    N_p=[];
    for j=1:S
        Q1 = n_aux*del_glob_t(p,j);
        Q2 = (sum(del_W_all(p,j,:))-del_W_all(p,j,p))./(C-1);
        Q3 = del_W_all(p,j,p);
        N_p_t = (Q1-N_k*Q2)./(Q3-Q2);
        
        if( ~isinf(N_p_t)&& ~isnan(N_p_t))
            N_p = [N_p;N_p_t];
        end
        
    end
     if(~isempty(N_p))     
     p_est(p) = mean(N_p)./N_k;
     else
         p_est(p) = 1./C; %unifom
     end
end

end

