function [sigma_all] = calculate_softlables_multiclass(net,X_aux_all,Y_aux_all,C)
%W: set of parameters/weights 
%   Detailed explanation goes here
sigma_all=[];
for cc=0:C-1
    ll = find(Y_aux_all==num2str(cc));
    Y_aux_c = Y_aux_all(ll);
    X_aux_c = X_aux_all(:,:,:,ll);
sigma = predict(net,X_aux_c);
sigma_all=[sigma_all,mean(sigma',2)];
end


end

