function [W_local,net] = small_NN_train_local_US_Census(XTrain_10,YTrain_10,epochs,B,W_G,flag_plot,C)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
N = length(YTrain_10);
numFeatures = 104;

layers = [featureInputLayer(numFeatures,'normalization','zscore')
        fullyConnectedLayer(32)
        fullyConnectedLayer(16)
        fullyConnectedLayer(8)
        fullyConnectedLayer(C)
        softmaxLayer
        classificationLayer];
%% weight init:
layers(2,1).Weights = W_G{1,1}; %need to change this-these are just weights
layers(2,1).Bias = W_G{1,2}; %default

layers(3,1).Weights = W_G{1,3};
layers(3,1).Bias = W_G{1,4};


% last layer-let it be zero for now:
layers(4,1).Weights = W_G{1,5};
layers(4,1).Bias = W_G{1,6};

layers(5,1).Weights = W_G{1,7};
layers(5,1).Bias = W_G{1,8};

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

net = trainNetwork(XTrain_10,"salary",layers,options);

W_local = {net.Layers(2,1).Weights,net.Layers(2,1).Bias,net.Layers(3,1).Weights,net.Layers(3,1).Bias,...
   net.Layers(4,1).Weights,net.Layers(4,1).Bias,net.Layers(5,1).Weights,net.Layers(5,1).Bias};

end

