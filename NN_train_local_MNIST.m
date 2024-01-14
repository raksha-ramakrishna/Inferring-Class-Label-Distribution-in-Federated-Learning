function [W_local,net] = NN_train_local_MNIST(XTrain_10,YTrain_10,epochs,B,W_G,flag_plot)
%Local learning MNIST CNN
N = length(YTrain_10);
%B = min(100,N); %batch size 
layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(2,'WeightsInitializer','zeros' )
    softmaxLayer
    classificationLayer];

%% weight init:
layers(2,1).Weights = W_G{1,1}; %need to change this-these are just weights
layers(2,1).Bias = W_G{1,2}; %default

layers(6,1).Weights = W_G{1,3};
layers(6,1).Bias = W_G{1,4};

layers(10,1).Weights = W_G{1,5};
layers(10,1).Bias = W_G{1,6};

% last layer-let it be zero for now:
layers(13,1).Weights = W_G{1,7};
layers(13,1).Bias = W_G{1,8};

options = trainingOptions('sgdm', ...
    'L2Regularization',0,...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',epochs, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'MiniBatchSize',B);% ...
if(flag_plot==1)
    options = trainingOptions('sgdm', ...
        'L2Regularization',0,...
        'InitialLearnRate',0.01, ...
        'MaxEpochs',epochs, ...
        'Shuffle','every-epoch', ...
        'Verbose',false, ...
        'MiniBatchSize',N,...
        'Plots','training-progress');
    
end
Ylabels = categorical(YTrain_10,{'2','3'});
%Ylabels = categorical(YTrain_new,0:1);

net = trainNetwork(XTrain_10,Ylabels,layers,options);

W_local = {net.Layers(2,1).Weights,net.Layers(2,1).Bias,net.Layers(6,1).Weights,net.Layers(6,1).Bias,...
    net.Layers(10,1).Weights,net.Layers(10,1).Bias,net.Layers(13,1).Weights,net.Layers(13,1).Bias};
%close all;

end

