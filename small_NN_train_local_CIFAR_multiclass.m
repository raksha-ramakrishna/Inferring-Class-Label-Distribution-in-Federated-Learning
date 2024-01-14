function [W_local,net] = small_NN_train_local_CIFAR_multiclass(XTrain_10,YTrain_10,epochs,B,W_G,flag_plot,C)
%Local learning MNIST CNN
N = length(YTrain_10);
%B = min(100,N); %batch size
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
        fullyConnectedLayer(C)
        softmaxLayer
        classificationLayer];


%% weight init:
layers(2,1).Weights = W_G{1,1}; %need to change this-these are just weights
layers(2,1).Bias = W_G{1,2}; %default

layers(6,1).Weights = W_G{1,3};
layers(6,1).Bias = W_G{1,4};


% last layer-let it be zero for now:
layers(9,1).Weights = W_G{1,5};
layers(9,1).Bias = W_G{1,6};

layers(11,1).Weights = W_G{1,7};
layers(11,1).Bias = W_G{1,8};

options = trainingOptions('sgdm', 'Momentum',0,...
    'L2Regularization',0,...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',epochs, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'MiniBatchSize',B);% ...

if(flag_plot==1)
    options = trainingOptions('sgdm','Momentum',0, ...
        'L2Regularization',0,...
        'InitialLearnRate',0.01, ...
        'MaxEpochs',epochs, ...
        'Shuffle','every-epoch', ...
        'Verbose',false, ...
        'MiniBatchSize',B,...
        'Plots','training-progress');
    
end
Ylabels = categorical(YTrain_10);
%Ylabels = categorical(YTrain_new,0:1);

net = trainNetwork(XTrain_10,Ylabels,layers,options);

W_local = {net.Layers(2,1).Weights,net.Layers(2,1).Bias,net.Layers(6,1).Weights,net.Layers(6,1).Bias,...
   net.Layers(9,1).Weights,net.Layers(9,1).Bias,net.Layers(11,1).Weights,net.Layers(11,1).Bias};
 
%close all;

end

