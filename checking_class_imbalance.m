clear all;
close all;
clc;
%ip_filename = 'all_FL_comparison_epochsE5B200_CIFAR_100_sampled_corrected.mat';
%ip_filename = 'all_FL_comparison_epochsE5B200_CIFAR_10_sampled.mat';
ip_filename = 'new_comparison_US_Census_sampling.mat';
load(ip_filename);
t_n = 10; %iteration of interest
K=10;
FL=10;
N = K*FL;
C = 2;
p_us = zeros(C,N);
p_them = zeros(C,N);
p_actual = zeros(C,N);

m=1; %index of our method to be chosen
for ff=1:FL
    for k=1:10
        tmp1 = SimplexProj(transpose(p_est_aux_grad(:,k,t_n,ff)));

        %SimplexProj(transpose(p_est_aux_grad(:,k,t_n,ff)));
        p_us(:,(ff-1)*K+k) = tmp1';
        
        tmp2 = SimplexProj(transpose(p_est_compare_mod(:,k,t_n,ff)));
        p_them(:,(ff-1)*K+k) = tmp2';
        
        p_actual(:,(ff-1)*K+k) = p_true(:,k,t_n,ff);
        
    end

end
% csvwrite('p_actual_CENSUS.csv',[p_us',p_actual']);
% csvwrite('p_them_CENSUS.csv',[p_them',p_actual']);
% csvwrite('p_us_CIFAR.csv',p_us');
% csvwrite('p_them_CIFAR.csv',p_them');
% csvwrite('p_actual_CIFAR.csv',p_actual');

%%
figure;
for i=1:C
subplot(1,2,i);
plot(p_actual(i,:),p_actual(i,:),'*');
hold on
plot(p_actual(i,:),p_us(i,:),'.');
plot(p_actual(i,:),p_them(i,:),'x','color','black');
grid on
end
legend('actual','aux grad','Wang et.al modified');
xlabel('p');
ylabel('$\hat{p}$');
