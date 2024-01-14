function [net] = weights_to_net(W_G,C)
%UNTITLED4 Summary of this function goes here


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



layers(1,1).Mean=[35.400002,0,0.10000000,0,0.69999999,0,0.10000000,0.10000000,0,137121.91,0,0,0,0,0,0,0.20000000,0,0.30000001,0.10000000,0.10000000,0,0.30000001,0,0,0,0.20000000,0,0.40000001,0,0.40000001,0,0,0,0,0.10000000,0,0,0,0,0.40000001,0,0.10000000,0,0.20000000,0.10000000,0.10000000,0.30000001,0.20000000,0.10000000,0.30000001,0.10000000,0,0,0.10000000,0,0,0.89999998,0.40000001,0.60000002,747.59998,0,41.500000,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0];
layers(1,1).StandardDeviation = [12.281693,0,0.30000001,0,0.45825756,0,0.30000001,0.30000001,0,83490.750,0,0,0,0,0,0,0.40000001,0,0.45825756,0.30000001,0.30000001,0,0.45825756,0,0,0,0.40000001,0,0.48989794,0,0.48989794,0,0,0,0,0.30000001,0,0,0,0,0.48989794,0,0.30000001,0,0.40000001,0.30000001,0.30000001,0.45825756,0.40000001,0.30000001,0.45825756,0.30000001,0,0,0.30000001,0,0,0.30000001,0.48989794,0.48989794,1600.2338,0,9.5000000,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];


net=assembleNetwork(layers);
end

