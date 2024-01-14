function [mean_KL,mean_MSE] = KL_divergence_and_MSE(ip_filename,op_filename,C,K,T_glob,fl)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
load(ip_filename);

errKL = zeros(K*fl,T_glob);
errKL_reg = zeros(K*fl,T_glob);
errKL_aux = zeros(K*fl,T_glob);
errKL_compare = zeros(K*fl,T_glob);
errKL_compare_mod = zeros(K*fl,T_glob);
errKL_aux_W= zeros(K*fl,T_glob);


for ff = 1:fl
    for t=1:T_glob
        our_est=[];
        wang_est=[];
        p_actual=[];
        for k=1:K
            p_actual = [p_actual,p_true(:,k,t,ff)];
            tmp1 = SimplexProj(transpose(p_est_org(:,k,t,ff)));
            our_est = [our_est,tmp1'];
            errKL_reg((ff-1)*K+k,t) = sum(p_true(:,k,t,ff) .* log((p_true(:,k,t,ff)+eps)./(tmp1'+eps)),1);
            tmp2 = SimplexProj(transpose(p_est_aux_grad(:,k,t,ff)));
            errKL_aux((ff-1)*K+k,t) = sum(p_true(:,k,t,ff) .* log((p_true(:,k,t,ff)+eps)./(tmp2'+eps)),1);
            tmp3 = SimplexProj(transpose(p_est_soft_label(:,k,t,ff)));
            errKL((ff-1)*K+k,t) = sum(p_true(:,k,t,ff) .* log((p_true(:,k,t,ff)+eps)./(tmp3'+eps)),1);
            tmp4 = SimplexProj(transpose(p_est_compare(:,k,t,ff)));
            %tmp4 = transpose(p_est_compare(:,k,t,ff)./sum(p_est_compare(:,k,t,ff)));
            errKL_compare((ff-1)*K+k,t) = sum(p_true(:,k,t,ff) .* log((p_true(:,k,t,ff)+eps)./(tmp4'+eps)),1);
            tmp5 = SimplexProj(transpose(p_est_compare_mod(:,k,t,ff)));
            wang_est = [wang_est,tmp5'];
            %tmp5 = transpose(p_est_compare_mod(:,k,t,ff)./sum(p_est_compare_mod(:,k,t,ff)));
            errKL_compare_mod((ff-1)*K+k,t) = sum(p_true(:,k,t,ff) .* log((p_true(:,k,t,ff)+eps)./(tmp5'+eps)),1);
            tmp6 = SimplexProj(transpose(p_est_aux_W_grad(:,k,t,ff)));
            errKL_aux_W((ff-1)*K+k,t) = sum(p_true(:,k,t,ff) .* log((p_true(:,k,t,ff)+eps)./(tmp6'+eps)),1);
            p_example = [tmp1',tmp3',tmp2',tmp4',tmp5',tmp6',p_true(:,k,t,ff)];
            %disp(tmp5);
            %             figure;
            %             stem(p_true(:,k,t,ff),'-*','color','black');
            %             hold on
            %             stem(tmp1);
            %             stem(tmp3);
            %             stem(tmp2);
            %             stem(tmp4);
            %             stem(tmp5);
            %             legend('actual distribution','initialized bias','soft-label estimator','aux grad estimator','Wang et.al','Wang et.al modified');
            %             grid on
            %
            %             close all;
            
        end
    end
end
%%
[pp,~]=size(errKL_reg);
for tt = 1:T_glob
    R = [errKL_reg(1:pp,tt),errKL(1:pp,tt),errKL_aux(1:pp,tt),errKL_compare(1:pp,tt),errKL_compare_mod(1:pp,tt),errKL_aux_W(1:pp,tt)];
    csvwrite(strcat('KL_',op_filename,num2str(tt-1),'.csv'),R);
end

mean_KL = [transpose(mean(errKL_reg)),transpose(mean(errKL)),transpose(mean(errKL_aux)),transpose(mean(errKL_compare)),transpose(mean(errKL_compare_mod)),transpose(mean(errKL_aux_W))];
%%
errMSE = zeros(K*fl,T_glob);
errMSE_reg = zeros(K*fl,T_glob);
errMSE_aux = zeros(K*fl,T_glob);
errMSE_compare = zeros(K*fl,T_glob);
errMSE_compare_mod = zeros(K*fl,T_glob);
errMSE_aux_W= zeros(K*fl,T_glob);

% p_est_aux_grad(isinf(p_est_aux_grad))=1;
% p_est_aux_grad(isnan(p_est_aux_grad))=0;
% p_est_compare(isinf(p_est_compare))=1;
% p_est_compare(isnan(p_est_compare))=1;

P_example = [];
C=1; %no change!
for ff = 1:fl
    for t=1:T_glob
        our_est=[];
        wang_est=[];
        p_actual=[];
        for k=1:K
            p_actual = [p_actual,p_true(:,k,t,ff)];
            tmp1 = SimplexProj(transpose(p_est_org(:,k,t,ff)));
            our_est = [our_est,tmp1'];
            errMSE_reg((ff-1)*K+k,t) = sum((p_true(:,k,t,ff)- tmp1').^2)./C;
            tmp2 = SimplexProj(transpose(p_est_aux_grad(:,k,t,ff)));
            errMSE_aux((ff-1)*K+k,t) = sum((p_true(:,k,t,ff)- tmp2').^2)./C;
            tmp3 = SimplexProj(transpose(p_est_soft_label(:,k,t,ff)));
            errMSE((ff-1)*K+k,t) = sum((p_true(:,k,t,ff)- tmp3').^2)./C;
            tmp4 = SimplexProj(transpose(p_est_compare(:,k,t,ff)));
            %tmp4 = transpose(p_est_compare(:,k,t,ff)./sum(p_est_compare(:,k,t,ff)));
            errMSE_compare((ff-1)*K+k,t) = sum((p_true(:,k,t,ff)- tmp4').^2)./C;
            tmp5 = SimplexProj(transpose(p_est_compare_mod(:,k,t,ff)));
            wang_est = [wang_est,tmp5'];
            %tmp5 = transpose(p_est_compare_mod(:,k,t,ff)./sum(p_est_compare_mod(:,k,t,ff)));
            errMSE_compare_mod((ff-1)*K+k,t) = sum((p_true(:,k,t,ff)- tmp5').^2)./C;
            tmp6 = SimplexProj(transpose(p_est_aux_W_grad(:,k,t,ff)));
            errMSE_aux_W((ff-1)*K+k,t) = sum((p_true(:,k,t,ff)- tmp6').^2)./C;
            p_example = [tmp1',tmp3',tmp2',tmp4',tmp5',tmp6',p_true(:,k,t,ff)];
            %disp(tmp5);
            %             figure;
            %             stem(p_true(:,k,t,ff),'-*','color','black');
            %             hold on
            %             stem(tmp1);
            %             stem(tmp3);
            %             stem(tmp2);
            %             stem(tmp4);
            %             stem(tmp5);
            %             legend('actual distribution','initialized bias','soft-label estimator','aux grad estimator','Wang et.al','Wang et.al modified');
            %             grid on
            %
            %             close all;
            
        end
    end
end
%%
[pp,~]=size(errMSE_reg);
for tt = 1:T_glob
    R = [errMSE_reg(1:pp,tt),errMSE(1:pp,tt),errMSE_aux(1:pp,tt),errMSE_compare(1:pp,tt),errMSE_compare_mod(1:pp,tt),errMSE_aux_W(1:pp,tt)];
    csvwrite(strcat('MSE_',op_filename,num2str(tt-1),'.csv'),R);
end

mean_MSE = [transpose(mean(errMSE_reg)),transpose(mean(errMSE)),transpose(mean(errMSE_aux)),transpose(mean(errMSE_aux_W)),transpose(mean(errMSE_compare)),transpose(mean(errMSE_compare_mod))];

end

