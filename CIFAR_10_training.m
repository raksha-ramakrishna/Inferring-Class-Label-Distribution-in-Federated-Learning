%% CIFAR-10
clear all;
close all;
clc;
%% load training data
oldpath = addpath(fullfile(matlabroot,'examples','deeplearning_shared','main'));
cifar10  = pwd;
%cifar10 = strcat(pwd,'\cifar-10-batches-mat');

[trainingImages,trainingLabels,testImages,testLabels] = helperCIFAR10Data.load(cifar10);

%w_error  = zeros(1,5); %no noise yet
%w_error = logspace(0,-3,5);
w_error = [logspace(1,-3,9),0];

%err1=zeros(numClasses*5,numClasses);
N = 1000; %some small number of samples.

err1=[];

for ww = 1:10
    est_ratio_all = [];
    true_ratio_all=[];
    
    for pp=1:100
        
       % N = randi([1000,40000],1);
        N_rand_indices = randperm(50000,N);
        
        XTrain_new = trainingImages(:,:,:,N_rand_indices);
        YTrain_new = trainingLabels(N_rand_indices);
        
        true_ratio = countcats(YTrain_new)./N;
        
        true_ratio_all = [true_ratio_all, true_ratio];
        
        
        %% These layers change for CIFAR-10 just because of training data:
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
            %fullyConnectedLayer(10,'WeightsInitializer',@(sz) rand(sz)*sqrt(w_error(ww)) )
            fullyConnectedLayer(10,'WeightsInitializer','he'); %default glorot
            softmaxLayer
            classificationLayer];
        
        %% weight init:
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
        
        
        del_H = h_new -h_0;
        
        CC = exp(h_0)./sum(exp(h_0));
        r = 0.01;
        C = 10;
        est_ratio = (del_H./(r))+ CC;
        
%         %% project est_ratio onto the probability simplex
%         est_ratio = transpose(SimplexProj(est_ratio'));
        
        est_ratio_all = [est_ratio_all,est_ratio];
        close all;
        disp(pp);
    end
    %filename = strcat('est_w_init_CIFAR_10_',num2str(ww),'.csv');
    filename = strcat('est_w_init_he_CIFAR_10_',num2str(ww),'.csv');
    %err1 = [err1;est_ratio_all-true_ratio_all];
    csvwrite(filename,est_ratio_all);
    filename = strcat('true_w_init_he_CIFAR_10_',num2str(ww),'.csv');
    %filename = strcat('true_w_init_CIFAR_10_',num2str(ww),'.csv');
    %err1 = [err1;est_ratio_all-true_ratio_all];
    csvwrite(filename,true_ratio_all);
    
end
% csvwrite('est_exact_inference_CIFAR_10.csv',est_ratio_all);
% csvwrite('true_exact_inference_CIFAR_10.csv',true_ratio_all);

errors_MSE = zeros(10,4);

for ww = 1:4
    errors_MSE(:,ww) = sum(err1((ww-1)*10+1:ww*10,:).^2,2)./100;
end


Gustaf Dahlander TR 31
Frances Hugle TR 33 