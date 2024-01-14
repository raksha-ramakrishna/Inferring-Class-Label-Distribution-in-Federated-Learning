%% CIFAR-10
clear all;
close all;
clc;
%% load training data
oldpath = addpath(fullfile(matlabroot,'examples','deeplearning_shared','main'));
cifar10  = pwd;
%cifar10 = strcat(pwd,'\cifar-10-batches-mat');

[XTrain,YTrain,XTest,YTest] = helperCIFAR10Data.load(cifar10);

%%
C=10; %number of classes
class_names = categories(YTest);
X_aux_all=[]; Y_aux_all=[];
for cc = 1:C
    
    indices_0t = find(YTest==class_names{cc});
    
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

T_glob=20;
epochs = 5; %local epochs
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
%%
b_updates=zeros(C,K,T_glob); % local bias updates
b_glob  = zeros(C,T_glob); %global bias updates

for fl=1:FL
    tic
    %     %% a particular setting of proportions:
    %     c_prop_ind  = randi([1,length(p_x)],K,1);
    % initial global weight:
    filterSize=[5 5];
    numFilters = 32;
    layers = [
        imageInputLayer([32 32 3])
        
        convolution2dLayer(filterSize,numFilters,'Padding',2)
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(3, 'Stride',2)
        % Repeat the 3 core layers to complete the middle of the network.
%         convolution2dLayer(filterSize,numFilters,'Padding',2)
%         reluLayer()
%         maxPooling2dLayer(3, 'Stride',2)
        
        %             convolution2dLayer(filterSize, numFilters,'Padding',2)
        %             reluLayer()
        %             maxPooling2dLayer(3,'Stride',2)
        
        convolution2dLayer(filterSize, numFilters,'Padding',2)
        reluLayer()
        maxPooling2dLayer(3,'Stride',2)
        
        fullyConnectedLayer(64)
        
        % Add an ReLU non-linearity.
        reluLayer
        
        % number of image categories are 10
        fullyConnectedLayer(10);%,'WeightsInitializer',@(sz) rand(sz)*sqrt(w_error(ww)) )
        softmaxLayer
        classificationLayer];
    
    % weight init:
    % initialize weights here -Glorot for all except last layer:
    FZ = 5*5*3; %size of filter times number of channels=1
    FZ_out  = 5*5*32; %size of filter times number of filters
    var1 = 2./(FZ + FZ_out); aa1 = sqrt(3*var1)./2;
    layers(2,1).Weights =  0.0001 * randn([filterSize 3 numFilters]); %%something from MATLAB online%unifrnd(-aa1,aa1,5,5,3,32); %need to change this-these are just weights
    layers(2,1).Bias = zeros(1,1,32); %default
    
    var3 = 2./(FZ*32 + FZ_out*32); aa3 = sqrt(3*var3)./2;
    layers(6,1).Weights = unifrnd(-aa3,aa3,5,5,32,32);
    layers(6,1).Bias = zeros(1,1,32);
    
    layers(9,1).Weights = unifrnd(-1,1,64,1568);
    layers(9,1).Bias = unifrnd(-1,1,64,1);
    layers(11,1).Weights = unifrnd(-1,1,C,64); 
    %layers(2).Weights = 0.0001 * randn([filterSize 3 numFilters]); %%something from MATLAB online
    layers(11,1).Bias = unifrnd(-1,1,C,1);
    
    W_G_0 = {layers(2,1).Weights,layers(2,1).Bias,layers(6,1).Weights,layers(6,1).Bias,...
        layers(9,1).Weights,layers(9,1).Bias,layers(11,1).Weights,layers(11,1).Bias};
    W_G = W_G_0;
    
    
    % this is to divide the data inot many clients
    N_all = zeros(K,1); %per client
    LB = 2000;%length(YTrain)./K; %how much data can each client have.(just 1000 max now)
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
            indices = find(YTrain((ii-1)*LB+1:ii*LB)==class_names{cc});
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
        b_glob(:,iter_glob) = W_G{1,8}; 
        
        W_local_all=[];
        [A_t,W_t,sigma_all] = small_eval_aux_data_multiclass_CIFAR(W_G,X_aux_all,Y_aux_all,epochs,0.01,C);% for the aux-gradient method
        for ii=1:K
            % take data from  client ii-divide info into K different chunks
            if(ii>1)
            randperm_L = randperm_L_all(N_all(ii-1)+1:N_all(ii-1)+N_all(ii)); 
            else
                randperm_L = randperm_L_all(1:N_all(1));
            end
            XTrain_10 = XTrain(:,:,:,randperm_L);
            YTrain_10 = YTrain(randperm_L);
           % flag_plot=1;
            [W_local,net] = small_NN_train_local_CIFAR_multiclass(XTrain_10,YTrain_10,epochs,B,W_G,flag_plot,C);
            %W_local_all(ii,1:8)=W_local;
            W_local_all = [W_local_all;W_local];
            
            b_updates(:,ii,iter_glob)= W_local{1,8};
            
            % regular method:
            p_est = regular_prop_est(W_G{1,8},W_local{1,8},epochs,B,N_all(ii),0.01);
            % aux grad method:
            p_aux = small_aux_data_grad_multiclass(W_G{1,8},W_local{1,8},A_t,C);
            p_est_aux_W = aux_data_grad_MNIST(W_G{1,8},W_local{1,8},A_t,W_t,W_G{1,7},W_local{1,7},C);
            %soft-label method:
            p_soft = small_soft_label_est_no_init_multiclass(net,X_aux_all,Y_aux_all,sigma_all,W_G{1,8},W_local{1,8},epochs,B,N_all(ii),0.01,C);
            %p_soft = aux_data_soft_label(W_G,W_local,A_t,epochs,0.01);
            %comparison
            p_comp = FL_class_imbalance_comparison_actual(W_t,W_G{1,7},W_local{1,7},C,10,N_all(ii));
            p_comp_mod = FL_class_imbalance_comparison_actual_mod(W_t,W_G{1,7},W_local{1,7},C,10,N_all(ii));

            % true value:
            p_org = countcats(YTrain_10)./length(YTrain_10);
            
            %% save values:
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

    disp(fl);
    
    save('all_FL_comparison_epochsE5B200_CIFAR_10_sampled.mat','p_true','p_est_org','p_est_aux_grad','p_est_aux_W_grad','p_est_compare','p_est_compare_mod','p_est_soft_label','epochs','B','T_glob');
    
    
end

%%
ip_filename = 'all_FL_comparison_epochsE5B200_CIFAR_10_sampled.mat';
op_filename = 'CIFAR_10_sampled';
[mean_KL,mean_MSE] = KL_divergence_and_MSE(ip_filename,op_filename,10,10,20,10);
%%
save('all_updatesCIFAR.mat','b_updates','b_glob','p_true','p_est_org','p_est_aux_grad','p_est_soft_label','epochs','B','T_glob');
save('all_FL_comparisonWU_new_soft_mult_epoch_and_new_aux_epochsE5B200_CIFAR_full_correctedB_compare_aux_E5.mat','p_true','p_est_org','p_est_aux_grad','p_est_aux_W_grad','p_est_soft_label','p_est_compare','p_est_compare_mod','epochs','B','T_glob');









%%
for ww = 1
    est_ratio_all = [];
    true_ratio_all=[];
    
    for pp=1:100
        
        N = randi([1000,40000],1);
        N_rand_indices = randperm(50000,N);
        
        XTrain_new = XTrain(:,:,:,N_rand_indices);
        YTrain_new = YTrain(N_rand_indices);
        
        true_ratio = countcats(YTrain_new)./N;
        
        true_ratio_all = [true_ratio_all, true_ratio];
        
        
        % These layers change for CIFAR-10 just because of training data:
        filterSize=[5 5];
        numFilters = 32;
        layers = [
            imageInputLayer([32 32 3])
            
            convolution2dLayer(filterSize,numFilters,'Padding',2)
            %batchNormalizationLayer
            reluLayer
            maxPooling2dLayer(3, 'Stride',2)
            % Repeat the 3 core layers to complete the middle of the network.
            convolution2dLayer(filterSize,numFilters,'Padding',2)
            reluLayer()
            maxPooling2dLayer(3, 'Stride',2)
            
            %             convolution2dLayer(filterSize, numFilters,'Padding',2)
            %             reluLayer()
            %             maxPooling2dLayer(3,'Stride',2)
            
            convolution2dLayer(filterSize, numFilters,'Padding',2)
            reluLayer()
            maxPooling2dLayer(3,'Stride',2)
            
            fullyConnectedLayer(64)
            
            % Add an ReLU non-linearity.
            reluLayer
            
            % number of image categories are 10
            fullyConnectedLayer(10)
            softmaxLayer
            classificationLayer];
        
        % weight init:
        %layers(13,1).Weights = sqrt(w_error(ww))*randn(1568,10);
        %layers(2).Weights = 0.0001 * randn([filterSize 3 numFilters]); %%something from MATLAB online
        layers(13,1).Bias = rand(10,1);
        
        h_0 = layers(13,1).Bias;
        
        options = trainingOptions('sgdm', ...
            'L2Regularization',0,...
            'InitialLearnRate',0.01, ...
            'MaxEpochs',1, ...
            'Shuffle','every-epoch', ...
            'Verbose',false, ...
            'MiniBatchSize',N);%, ...
        % 'Plots','training-progress');
        
        net = trainNetwork(XTrain_new,YTrain_new,layers,options);
        h_new = net.Layers(13,1).Bias;
        
        for iter = 1:20
            net = trainNetwork(XTrain_new,YTrain_new,net.Layers,options);
            h_new = [h_new,net.Layers(13,1).Bias];
        end
        
        del_H = h_new -h_0;
        
        CC = exp(h_0)./sum(exp(h_0));
        r = 0.01;
        C = 10;
        est_ratio = (del_H./(r))+ CC;
        
        %         %% project est_ratio onto the probability simplex
        %         est_ratio = transpose(SimplexProj(est_ratio'));
        
        est_ratio_all = [est_ratio_all;est_ratio];
        close all;
        disp(pp);
    end
    filename = strcat('est_iter_CIFAR_10','.csv');
    %err1 = [err1;est_ratio_all-true_ratio_all];
    csvwrite(filename,est_ratio_all);
    filename = strcat('true_iter_CIFAR_10','.csv');
    %err1 = [err1;est_ratio_all-true_ratio_all];
    csvwrite(filename,true_ratio_all);
    
end


