function [N_class,N_new] = sampling_for_each_client(N,C)
% N: number of samples overall
% C: number of classes 
% N_class: array-number of samples per class
%N_new: new total number of samples for a client

prob_simplex  = zeros(C,1); %fraction of samples per class

c_pts = rand(C-1,1); % C points in the interval 0 and 1
%c_pts = 1; %only for US Census 
all_vals = sort([0;c_pts;1],'ascend');
prob_simplex = diff(all_vals);
% for i=1:C
%     if(i<C)
%     prob_simplex(i) = rand(1);
%     else
%         prob_simplex(i) =  1-sum(prob_simplex(1:i-1));
%     end
% end

N_class = round(N.*prob_simplex);
N_new = sum(N_class); 
end

