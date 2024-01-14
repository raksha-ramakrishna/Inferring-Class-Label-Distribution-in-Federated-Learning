%% comparisons for methods-multiclass (10-class)
%% compare all proposed methods for W initialized to zero.
clear all;
close all;
clc;
%% load the MNIST data here:
oldpath = addpath(fullfile(matlabroot,'examples','nnet','main'));
filenameImagesTrain = 'train-images.idx3-ubyte';
filenameLabelsTrain = 'train-labels.idx1-ubyte';
filenameImagesTest = 't10k-images.idx3-ubyte';
filenameLabelsTest = 't10k-labels.idx1-ubyte';

XTrain = processImagesMNIST(filenameImagesTrain);
YTrain = processLabelsMNIST(filenameLabelsTrain);


%%  choose all: there are about 5000 of each.
%% choose the aux dataset here:
% just 10 data points -same data as class imbalance
C=10; %number of classes

XTest = processImagesMNIST(filenameImagesTest);
YTest = processLabelsMNIST(filenameLabelsTest);

X_aux_all=[]; Y_aux_all=[];

for cc = 0:C-1
    
    indices_0t = find(YTest==num2str(cc));
    
    XTest_0 = XTest(:,:,:,indices_0t);
    YTest_0 = YTest(indices_0t);
    
    aux_data_num=10;
    ind_aux_0 = randi([1,length(YTest_0)],aux_data_num,1);
    
    X_aux_0 = XTest_0(:,:,:,ind_aux_0);
    Y_aux_0 = YTest_0(ind_aux_0);
    
    X_aux_all = cat(4,X_aux_all,X_aux_0);
    Y_aux_all = [Y_aux_all;Y_aux_0];
end
%%
K = 10; % 10 clients-have different FL runs with different proportions
% p_x    = linspace(0,1,11);
% load('client_indices_MNIST.mat');
T_glob=1;
epochs = 1; %local epochs
B  = 256; %mini-batch size
flag_plot=0;

%%
FL = 10;%different iterations of FL
p_true = zeros(C,K,T_glob,FL);
p_est_org = zeros(C,K,T_glob,FL);
p_est_aux_grad = zeros(C,K,T_glob,FL);
p_est_aux_W_grad  = zeros(C,K,T_glob,FL);
p_est_soft_label = zeros(C,K,T_glob,FL);
p_est_compare = zeros(C,K,T_glob,FL);
p_est_compare_mod = zeros(C,K,T_glob,FL);
%
b_updates=zeros(C,K,T_glob); % local bias updates
b_glob  = zeros(C,T_glob); %global bias updates

%load('all_FL_comparison_MNIST_aux_changedE5_sampling.mat');

for fl=1:FL
    tic
    %     %% a particular setting of proportions:
    %     c_prop_ind  = randi([1,length(p_x)],K,1);
    % initial global weight:
    layers = [
        imageInputLayer([28 28 1])
        
        convolution2dLayer(3,8,'Padding','same')
        batchNormalizationLayer
        reluLayer
        
        maxPooling2dLayer(2,'Stride',2)
        
        %         convolution2dLayer(3,16,'Padding','same')
        %         batchNormalizationLayer
        %         reluLayer
        %
        %         maxPooling2dLayer(2,'Stride',2)
        
        convolution2dLayer(3,16,'Padding','same')
        batchNormalizationLayer
        reluLayer
        
        fullyConnectedLayer(C,'WeightsInitializer','zeros' )
        softmaxLayer
        classificationLayer];
    
    % initialize weights here -Glorot for all except last layer:
    layers(1,1).Mean = 0.3;
    
    layers(3,1).TrainedVariance = 0.002*ones(1,1,8);
    layers(3,1).TrainedMean  = 0.001*ones(1,1,8);
    layers(3,1).Offset = zeros(1,1,8);
    layers(3,1).Scale = ones(1,1,8);
    
    layers(7,1).TrainedVariance = 0.02*ones(1,1,16);
    layers(7,1).TrainedMean  = 0.01*ones(1,1,16);
    layers(7,1).Offset = zeros(1,1,16);
    layers(7,1).Scale = ones(1,1,16);
    
    FZ = 3*3; %size of filter times number of channels=1
    FZ_out  = 3*3; %size of filter times number of filters
    var1 = 2./(FZ + FZ_out*8); aa1 = sqrt(3*var1)./2;
    layers(2,1).Weights = unifrnd(-aa1,aa1,3,3,1,8); %need to change this-these are just weights
    layers(2,1).Bias = zeros(1,1,8); %default
    
    %     var2  = 2./(FZ*8 + FZ_out*16); aa2 = sqrt(3*var2)./2;
    %     layers(6,1).Weights = unifrnd(-aa2,aa2,3,3,8,16);
    %     layers(6,1).Bias = zeros(1,1,16);
    
    var3 = 2./(FZ*16 + FZ_out*16); aa3 = sqrt(3*var3)./2;
    layers(6,1).Weights = unifrnd(-aa3,aa3,3,3,8,16);
    layers(6,1).Bias = zeros(1,1,16);
    
    % last layer-let it be zero for now:
    %layers(9,1).Weights = unifrnd(-1,1,C,3136);
    layers(9,1).Weights = zeros(C,3136);
    %layers(13,1).Weights = zeros(2,1568);
    layers(9,1).Bias = unifrnd(-1,1,C,1);
    
    % Global weight initial:
    W_G_0 = {layers(2,1).Weights,layers(2,1).Bias,layers(6,1).Weights,layers(6,1).Bias,...
        layers(9,1).Weights,layers(9,1).Bias};
    W_G = W_G_0;
    
    N_all = zeros(K,1); %per client
    LB = 3000;%length(YTrain)./K; %how much data can each client have.
    randperm_L_all = [];
    %     for ii=1:K
    %         N_all(ii) = randi([100,LB/2],1); %size is between 1 and 3000
    %         randperm_L = randi([(ii-1)*LB+1,ii*LB],N_all(ii),1);
    %         randperm_L_all = [randperm_L_all;randperm_L];
    %     end
    for ii=1:K
        N_all(ii) = randi([100,LB/2],1); %size is between 1 and 3000
        [N_class,N_new]=sampling_for_each_client(N_all(ii),C);
        randperm_L=[];
        n_c=[];
        client_indices = transpose((ii-1)*LB+1:ii*LB);
        for cc=1:C
            indices = find(YTrain((ii-1)*LB+1:ii*LB)==num2str(cc));
            if(length(indices)>N_class(cc))
                ind_cc = randperm(length(indices),N_class(cc));
                n_c = [n_c;N_class(cc)];
            else
                ind_cc = 1:length(indices);
                n_c = [n_c;length(indices)];
            end
            randperm_L = [randperm_L;client_indices(indices(ind_cc))];
        end
        N_all(ii) = sum(n_c);
        % randperm_L = randi([(ii-1)*LB+1,ii*LB],N_all(ii),1);
        randperm_L_all = [randperm_L_all;randperm_L];
    end
    
    for iter_glob = 1:T_glob
        % W_local_all=cell(K,8);
        b_glob(:,iter_glob) = W_G{1,6};
        
        W_local_all=[];
        net1 = assembleNetwork(layers);
        [A_t,W_t,sigma_all] = small_eval_aux_data_multiclass(W_G,X_aux_all,Y_aux_all,epochs,0.01,C,net1);% for the aux-gradient method
        for ii=1:K
            % take data from  client ii-divide info into K different chunks
            if(ii>1)
                randperm_L = randperm_L_all(sum(N_all(1:ii-1))+1:sum(N_all(1:ii)));
            else
                randperm_L = randperm_L_all(1:N_all(1));
            end
            XTrain_10 = XTrain(:,:,:,randperm_L);
            YTrain_10 = YTrain(randperm_L);
            
            [W_local,net] = small_NN_train_local_MNIST_multiclass(XTrain_10,YTrain_10,epochs,N_all(ii),W_G,flag_plot,C);
            %W_local_all(ii,1:8)=W_local;
            b_updates(:,ii,iter_glob)= W_local{1,6};
            W_local_all = [W_local_all;W_local];
            % regular method:
            p_est = regular_prop_est(W_G{1,6},W_local{1,6},epochs,N_all(ii),N_all(ii),0.01);
            % aux grad method:
            p_aux = small_aux_data_grad_multiclass(W_G{1,6},W_local{1,6},A_t,C);
            p_est_aux_W = aux_data_grad_MNIST(W_G{1,6},W_local{1,6},A_t,W_t,W_G{1,5},W_local{1,5},C);
            %soft-label method:
            %p_soft = small_soft_label_est_multiclass(net,X_aux_all,Y_aux_all,W_G{1,6},W_local{1,6},epochs,B,N_all(ii),0.01,C);
            p_soft = small_soft_label_est_no_init_multiclass(net,X_aux_all,Y_aux_all,sigma_all,W_G{1,6},W_local{1,6},epochs,B,N_all(ii),0.01,C);
            %comparison
            p_comp = FL_class_imbalance_comparison_actual(W_t,W_G{1,5},W_local{1,5},C,10,N_all(ii));
            p_comp_mod = FL_class_imbalance_comparison_actual_mod(W_t,W_G{1,5},W_local{1,5},C,10,N_all(ii));
            
            %p_soft = aux_data_soft_label(W_G,W_local,A_t,epochs,0.01);
            % true value:
            p_org = countcats(YTrain_10)./length(YTrain_10);
            
            % save values:
            p_true(:,ii,iter_glob,fl) = p_org; %100 different iterations of FL
            p_est_org(:,ii,iter_glob,fl) = p_est;
            p_est_aux_grad(:,ii,iter_glob,fl) = p_aux;
            p_est_aux_W_grad(:,ii,iter_glob,fl) = p_est_aux_W;
            p_est_soft_label(:,ii,iter_glob,fl) = p_soft;
            p_est_compare(:,ii,iter_glob,fl) = p_comp;
            p_est_compare_mod(:,ii,iter_glob,fl) = p_comp_mod;
        end
        N_full = sum(sum(N_all));
        for jj=1:length(W_G_0)
            
            TRR=zeros(size(W_local_all{1,jj}));
            for ii=1:K %client number:
                %TRR = TRR+(1/K).*W_local_all{ii,jj};
                TRR = TRR+((N_all(ii))./N_full)*W_local_all{ii,jj};
            end
            W_G{1,jj} = TRR;
        end
        
    end
    toc
    disp(fl)
    save('MNIST_exact.mat','p_true','p_est_org','p_est_aux_grad','p_est_aux_W_grad','p_est_soft_label','p_est_compare','p_est_compare_mod','epochs','B','T_glob');
    
end
%%

ip_filename='MNIST_exact.mat';
op_filename = 'MNIST_exact';
[mean_KL,mean_MSE] = KL_divergence_and_MSE(ip_filename,op_filename,10,10,T_glob,FL);


% save('all_FL_comparisonWU_new_soft_mult_epoch_and_new_aux_epochsE5B200_MNIST_full_corrected_comp_actual_aux_changedE1.mat','p_true','p_est_org','p_est_aux_grad','p_est_aux_W_grad','p_est_soft_label','p_est_compare','p_est_compare_mod','epochs','B','T_glob');
% 
% save('all_updates_MNIST.mat','b_updates','b_glob','p_true','p_est_org','p_est_aux_grad','p_est_soft_label','epochs','B','T_glob');


%save('all_FL_comparisonWU_new_soft_and_aux_epochsE2B200_MNIST_full.mat','p_true','p_est_org','p_est_aux_grad','p_est_soft_label','epochs','B','T_glob');

% save('all_FL_comparisonWU_new_soft_and_aux_epochsE2B200_MNIST.mat','p_true','p_est_org','p_est_aux_grad','p_est_soft_label','epochs','B','T_glob');
%save('all_FL_comparisonWU35T10E5_new_soft.mat','p_true','p_est_org','p_est_aux_grad','p_est_soft_label','B','epochs','T_glob');
%%
load('all_FL_comparison_MNIST_aux_changedE5_sampling.mat');
errKL = zeros(K*fl,T_glob);
errKL_reg = zeros(K*fl,T_glob);
errKL_aux = zeros(K*fl,T_glob);
errKL_compare = zeros(K*fl,T_glob);
errKL_compare_mod = zeros(K*fl,T_glob);
errKL_aux_W= zeros(K*fl,T_glob);

% p_est_aux_grad(isinf(p_est_aux_grad))=1;
% p_est_aux_grad(isnan(p_est_aux_grad))=0;
% p_est_compare(isinf(p_est_compare))=1;
% p_est_compare(isnan(p_est_compare))=1;

P_example = [];
our_est=[];
wang_est=[];
p_actual=[];
for ff = 1:fl
    for t=1:T_glob
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
pp=100;
for tt = 1:T_glob
    R = [errKL_reg(1:pp,tt),errKL(1:pp,tt),errKL_aux(1:pp,tt),errKL_compare(1:pp,tt),errKL_compare_mod(1:pp,tt),errKL_aux_W(1:pp,tt)];
    csvwrite(strcat('KL_MNIST',num2str(tt-1),'.csv'),R);
end







%%
fl_no = 1;
figure;
title(strcat('Local epochs=',num2str(epochs)))
for ii=1:K
    subplot(2,5,ii);
    plot(vec(p_est_org(1,ii,1:T_glob,fl_no)),'-o');
    hold on
    plot(vec(p_est_aux_grad(1,ii,1:T_glob,fl_no)),'-o');
    plot(vec(p_est_soft_label(1,ii,1:T_glob,fl_no)),'-o');
    plot(vec(p_true(1,ii,1:T_glob,fl_no)),'-*','color','black');
    grid on
    if(ii==5)
        legend('estimated (org)','aux grad','soft label','true');
    end
    ylabel('p');
    xlabel('global iter k');
    title(strcat('client',num2str(ii)));
end
%plot(MSE_soft_label,'-o');