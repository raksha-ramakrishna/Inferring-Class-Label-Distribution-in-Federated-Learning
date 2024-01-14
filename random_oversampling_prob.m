function [N_class,N_new] = random_oversampling_prob(N,C)
%only for 2 class:
%
C=2;
k = randi([0,1],1);
c_pt = k*0.2*rand(1)+(1-k)*(0.8+0.2*rand(1)); 
prob_simplex   = [c_pt;1-c_pt];
N_class = round(N.*prob_simplex);
N_new = sum(N_class); 
end

