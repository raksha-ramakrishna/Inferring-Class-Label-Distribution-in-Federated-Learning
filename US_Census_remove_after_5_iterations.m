%% US Census case with no change:
clear all;
close all;
clc;
load census1994.mat
%%
% Education  number and level consolidated:
edOrder = unique(adultdata.education_num,"stable");
edCats = unique(adultdata.education,"stable");
[~,edIdx] = sort(edOrder);

adultdata.education = categorical(adultdata.education, ...
    edCats(edIdx),"Ordinal",true);
adultdata.education_num = [];

adulttest.education = categorical(adulttest.education, ...
    edCats(edIdx),"Ordinal",true);
adulttest.education_num = [];



% converting categorical input to one-hot encoded vectors
categoricalInputNames = ["workClass","education","marital_status","occupation","relationship","race","sex","native_country"];
tblTrain = convertvars(adultdata,categoricalInputNames,'categorical');

name = "native_country";
nativecountries = table2array(unique(tblTrain(:,name)));
nativecountries = string(nativecountries(1:41));
for i = 1:numel(categoricalInputNames)
    name = categoricalInputNames(i);
    oh = onehotencode(tblTrain(:,name));
    tblTrain = addvars(tblTrain,oh,'After',name);
    tblTrain(:,name) = [];
end



tblTrain = splitvars(tblTrain);
XTrain  = rmmissing(tblTrain);

numFeatures = size(tblTrain,2) - 1;


YTrain = table2array(XTrain(:,105));
%tblLabel = table2array(adultdata(:,end));
%tblLabel   = labels(trainingIndices);

tblTest = convertvars(adulttest,categoricalInputNames,'categorical');

for i = 1:numel(categoricalInputNames)
    name = categoricalInputNames(i);
    if(i==8)
        oh = onehotencode(tblTest(:,name),"ClassNames",nativecountries);
    else
        oh = onehotencode(tblTest(:,name));
    end
    tblTest = addvars(tblTest,oh,'After',name);
    tblTest(:,name) = [];
end

tblTest = splitvars(tblTest);
XTest  = rmmissing(tblTest);

YTest = table2array(XTest(:,105));
%% validation/test data:
XVal = XTest(1:100,:);
YVal = YTest(1:100);
%% Auxilliary data:
C=2;
load('aux_data_for_US_census.mat');
K = 50; % 10 clients-have different FL runs with different proportions

T_glob=200;
epochs = 5; %local epochs
B  = 256; %mini-batch size
flag_plot=0;

p_true = zeros(C,K,T_glob);
p_est_org = zeros(C,K,T_glob);
p_est_aux_grad = zeros(C,K,T_glob);
p_est_aux_W_grad  = zeros(C,K,T_glob);
p_est_soft_label = zeros(C,K,T_glob);
p_est_compare = zeros(C,K,T_glob);
p_est_compare_mod = zeros(C,K,T_glob);
%%
b_updates=zeros(C,K,T_glob); % local bias updates
b_glob  = zeros(C,T_glob); %global bias updates

load('layers_US_census.mat');

% tic
% layers = [featureInputLayer(numFeatures,'normalization','zscore')
%     fullyConnectedLayer(32)
%     fullyConnectedLayer(16)
%     fullyConnectedLayer(8)
%     fullyConnectedLayer(C)
%     softmaxLayer
%     classificationLayer];
% 
% layers(1,1).Mean=[35.400002,0,0.10000000,0,0.69999999,0,0.10000000,0.10000000,0,137121.91,0,0,0,0,0,0,0.20000000,0,0.30000001,0.10000000,0.10000000,0,0.30000001,0,0,0,0.20000000,0,0.40000001,0,0.40000001,0,0,0,0,0.10000000,0,0,0,0,0.40000001,0,0.10000000,0,0.20000000,0.10000000,0.10000000,0.30000001,0.20000000,0.10000000,0.30000001,0.10000000,0,0,0.10000000,0,0,0.89999998,0.40000001,0.60000002,747.59998,0,41.500000,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0];
% layers(1,1).StandardDeviation = [12.281693,0,0.30000001,0,0.45825756,0,0.30000001,0.30000001,0,83490.750,0,0,0,0,0,0,0.40000001,0,0.45825756,0.30000001,0.30000001,0,0.45825756,0,0,0,0.40000001,0,0.48989794,0,0.48989794,0,0,0,0,0.30000001,0,0,0,0,0.48989794,0,0.30000001,0,0.40000001,0.30000001,0.30000001,0.45825756,0.40000001,0.30000001,0.45825756,0.30000001,0,0,0.30000001,0,0,0.30000001,0.48989794,0.48989794,1600.2338,0,9.5000000,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
% layers(2,1).Weights = unifrnd(-1,1,32,numFeatures);
% layers(2,1).Bias = unifrnd(-1,1,32,1);
% 
% layers(3,1).Weights = unifrnd(-1,1,16,32);
% layers(3,1).Bias = unifrnd(-1,1,16,1);
% 
% layers(4,1).Weights = unifrnd(-1,1,8,16);
% layers(4,1).Bias = unifrnd(-1,1,8,1);
% 
% % layers(5,1).Weights = unifrnd(-1,1,2,8);
% layers(5,1).Weights = zeros(2,8);
% layers(5,1).Bias = unifrnd(-1,1,2,1);

W_G_0 = {layers(2,1).Weights,layers(2,1).Bias,layers(3,1).Weights,layers(3,1).Bias,...
    layers(4,1).Weights,layers(4,1).Bias,layers(5,1).Weights,layers(5,1).Bias};
W_G = W_G_0;
%% client selection:
load('data_for_US_census.mat');
cross_entropy_cost = zeros(K,T_glob+1);
accuracy = zeros(T_glob+1,1);
removed_clients=[]; % indices of clients that were removed
K_all=1:K;
for iter_glob = 1:T_glob
    % W_local_all=cell(K,8);
    
    % this is to divide the data into many clients-change for every
    % iteration
    b_glob(:,iter_glob) = W_G{1,8};
    
    W_local_all=[];
    
%             if(iter_glob==1)
%                 net1=assembleNetwork(layers);
%     
%             else
%                 net1=assembleNetwork(net.Layers);
%             end
%     %         [A_t,W_t,sigma_all] = small_eval_aux_data_US_Census(W_G,X_aux_all,Y_aux_all,epochs,0.01,C,net1);% for the aux-gradient method
    %tic
    
    
    
    
    
    for ii=1:K
        if(ii==removed_clients)
            continue
        end
        % take data from  client ii-divide info into K different chunks
        if(ii>1)
            randperm_L = randperm_L_all(sum(N_all(1:ii-1))+1:sum(N_all(1:ii)));
        else
            randperm_L = randperm_L_all(1:N_all(1));
        end
        XTrain_10 = XTrain(randperm_L,:);
        YTrain_10 = YTrain(randperm_L);
        
%         if(iter_glob==1)
%                 est_label = predict(net1,XTrain_10);
%                 cross_entropy_cost(ii,1) = crossentropy(est_label,onehotencode(YTrain_10,C));
%         end
        % flag_plot=1;
        [W_local,net] = small_NN_train_local_US_Census(XTrain_10,YTrain_10,epochs,B,W_G,flag_plot,C);
        est_label = predict(net,XTrain_10);
        cross_entropy_cost(ii,iter_glob) = crossentropy(est_label,onehotencode(YTrain_10,C));
        %W_local_all(ii,1:8)=W_local;
        W_local_all = [W_local_all;W_local];
        
        b_updates(:,ii,iter_glob)= W_local{1,8};
        
        % regular method:
        p_est = regular_prop_est(W_G{1,8},W_local{1,8},epochs,B,N_all(ii),0.01);
        
        % aux grad method:
        %             p_aux = small_aux_data_grad_multiclass(W_G{1,8},W_local{1,8},A_t,C);
        %             p_est_aux_W = aux_data_grad_MNIST(W_G{1,8},W_local{1,8},A_t,W_t,W_G{1,7},W_local{1,7},C);
        %soft-label method:
        %            p_soft = small_soft_label_est_no_init_multiclass(net,X_aux_all,Y_aux_all,sigma_all,W_G{1,8},W_local{1,8},epochs,B,N_all(ii),0.01,C);
        %p_soft = aux_data_soft_label(W_G,W_local,A_t,epochs,0.01);
        %comparison:
        %            p_comp = FL_class_imbalance_comparison_actual(W_t,W_G{1,7},W_local{1,7},C,10,N_all(ii));
        %            p_comp_mod = FL_class_imbalance_comparison_actual_mod(W_t,W_G{1,7},W_local{1,7},C,10,N_all(ii));
        % true value:
        p_org = countcats(YTrain_10)./length(YTrain_10);
        
        %% save values:
        p_true(:,ii,iter_glob) = p_org; %100 different iterations of FL
        p_est_org(:,ii,iter_glob) = p_est;
        %             p_est_aux_grad(:,ii,iter_glob) = p_aux;
        %             p_est_aux_W_grad(:,ii,iter_glob) = p_est_aux_W;
        %             p_est_soft_label(:,ii,iter_glob) = p_soft;
        %             p_est_compare(:,ii,iter_glob) = p_comp;
        %             p_est_compare_mod(:,ii,iter_glob) = p_comp_mod;
        
        if(iter_glob==10 && (p_est(1)<0.15 || p_est(2)<0.15))
            removed_clients = [removed_clients,ii];
        end
    end
    if(iter_glob==10)
        K_all(removed_clients)=[]; % remove those clients
    end
    %toc
    N_full = sum(sum(N_all))-sum(N_all(removed_clients));
    
    for jj=1:length(W_G_0)
        
        TRR=zeros(size(W_local_all{1,jj}));
        for ii=K_all %client number:
            %TRR = TRR+(1/K).*W_local_all{ii,jj};
            TRR = TRR+((N_all(ii))./N_full)*W_local_all{ii,jj};
        end
        W_G{1,jj} = TRR;
    end
     net1 = weights_to_net(W_G,C);
        predicted_labels = predict(net1,XVal);
        %accuracy(iter_glob) = crossentropy(predicted_labels,onehotencode(YVal,C));
        [~,~,~,accuracy(iter_glob)] =perfcurve(YVal,predicted_labels(:,1),'<=50K');
    
    
    
    disp(iter_glob);
    
end
toc
figure;
plot(sum(cross_entropy_cost(:,1:iter_glob)),'-o');
save('US_census_address_class_imbalance_remove_iter_10_change_only_p_less_clients_200_iter_acc.mat','cross_entropy_cost','accuracy','p_true','p_est_org','p_est_aux_grad','p_est_soft_label','p_est_compare','removed_clients','p_est_compare_mod','p_est_aux_W_grad','epochs','B','T_glob','b_updates','b_glob');

