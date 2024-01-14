function [mean_MSE_1,mean_MSE_2] = KL_divergence_and_MSE_oversample(ip_filename,op_filename,C,K,T_glob,fl)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
load(ip_filename);

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


C=1; %no change!
for gg=1:2
    
    for t=1:T_glob
        p_example = [];
        for ff = 1:fl
            
            our_est=[];
            wang_est=[];
            p_actual=[];
            for k=1:K
                p_actual = [p_actual,p_true(:,k,t,ff,1)];
                tmp1 = SimplexProj(transpose(p_est_org(:,k,t,ff,gg)));
                our_est = [our_est,tmp1'];
                errMSE_reg((ff-1)*K+k,t) = sum((p_true(:,k,t,ff,1)- tmp1').^2)./C;
                tmp2 = SimplexProj(transpose(p_est_aux_grad(:,k,t,ff,gg)));
                errMSE_aux((ff-1)*K+k,t) = sum((p_true(:,k,t,ff,1)- tmp2').^2)./C;
                tmp3 = SimplexProj(transpose(p_est_soft_label(:,k,t,ff,gg)));
                errMSE((ff-1)*K+k,t) = sum((p_true(:,k,t,ff,1)- tmp3').^2)./C;
                tmp4 = SimplexProj(transpose(p_est_compare(:,k,t,ff,gg)));
                %tmp4 = transpose(p_est_compare(:,k,t,ff)./sum(p_est_compare(:,k,t,ff)));
                errMSE_compare((ff-1)*K+k,t) = sum((p_true(:,k,t,ff,1)- tmp4').^2)./C;
                tmp5 = SimplexProj(transpose(p_est_compare_mod(:,k,t,ff,gg)));
                wang_est = [wang_est,tmp5'];
                %tmp5 = transpose(p_est_compare_mod(:,k,t,ff)./sum(p_est_compare_mod(:,k,t,ff)));
                errMSE_compare_mod((ff-1)*K+k,t) = sum((p_true(:,k,t,ff,1)- tmp5').^2)./C;
                tmp6 = SimplexProj(transpose(p_est_aux_W_grad(:,k,t,ff,gg)));
                errMSE_aux_W((ff-1)*K+k,t) = sum((p_true(:,k,t,ff,1)- tmp6').^2)./C;
                estimators = [tmp1',tmp3',tmp2',tmp4',tmp5',tmp6'];
                p_example = [p_example;[estimators(1,:),p_true(1,k,t,ff,1)]];
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
    pp=length(errMSE_reg);
    for tt = 1:T_glob
        R = [errMSE_reg(1:pp,tt),errMSE(1:pp,tt),errMSE_aux(1:pp,tt),errMSE_compare(1:pp,tt),errMSE_compare_mod(1:pp,tt),errMSE_aux_W(1:pp,tt)];
        csvwrite(strcat('MSE_',num2str(gg),op_filename,num2str(tt-1),'.csv'),R);
    end
    if(gg==1)
        mean_MSE_1 = [transpose(mean(errMSE_reg)),transpose(mean(errMSE)),transpose(mean(errMSE_aux)),transpose(mean(errMSE_aux_W)),transpose(mean(errMSE_compare)),transpose(mean(errMSE_compare_mod))];
    else
        mean_MSE_2 = [transpose(mean(errMSE_reg)),transpose(mean(errMSE)),transpose(mean(errMSE_aux)),transpose(mean(errMSE_aux_W)),transpose(mean(errMSE_compare)),transpose(mean(errMSE_compare_mod))];
        
    end
end

