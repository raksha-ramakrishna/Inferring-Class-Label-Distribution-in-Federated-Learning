%%US Census Income
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

%% Auxilliary data using test data:
C=2;
class_names = categories(YTest);
X_aux_all=[]; Y_aux_all=[];
for cc = 1:C
    
    indices_0t = find(YTest==class_names{cc});
    
    XTest_0 = XTest(indices_0t,:);
    YTest_0 = YTest(indices_0t);
    
    aux_data_num=10;
    ind_aux_0 = randi([1,length(YTest_0)],aux_data_num,1);
    
    X_aux_0 = XTest_0(ind_aux_0,:);
    Y_aux_0 = YTest_0(ind_aux_0);
    
    X_aux_all = [X_aux_all;X_aux_0];
    Y_aux_all = [Y_aux_all;Y_aux_0];
end

%%
K = 10; % 10 clients-have different FL runs with different proportions

T_glob=30;
epochs = 10; %local epochs
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
    layers = [featureInputLayer(numFeatures,'normalization','zscore')
        fullyConnectedLayer(32)
        fullyConnectedLayer(16)
        fullyConnectedLayer(8)
        fullyConnectedLayer(C)
        softmaxLayer
        classificationLayer];
    
    layers(2,1).Weights = unifrnd(-1,1,32,numFeatures);
    layers(2,1).Bias = unifrnd(-1,1,32,1);
    
    layers(3,1).Weights = unifrnd(-1,1,16,32);
    layers(3,1).Bias = unifrnd(-1,1,16,1);
    
    layers(4,1).Weights = unifrnd(-1,1,8,16);
    layers(4,1).Bias = unifrnd(-1,1,8,1);
    
    layers(5,1).Weights = unifrnd(-1,1,2,8);
    layers(5,1).Bias = unifrnd(-1,1,2,1);
    
    W_G_0 = {layers(2,1).Weights,layers(2,1).Bias,layers(3,1).Weights,layers(3,1).Bias,...
        layers(4,1).Weights,layers(4,1).Bias,layers(5,1).Weights,layers(5,1).Bias};
    W_G = W_G_0;
    
    % this is to divide the data into many clients
    N_all = zeros(K,1); %per client
    LB = floor(length(YTrain)./K); %how much data can each client have.
    randperm_L_all = [];
    
    for ii=1:K
        N_all(ii) = randi([100,LB/2],1); %size is between 1 and 3000
        randperm_L = randi([(ii-1)*LB+1,ii*LB],N_all(ii),1);
        randperm_L_all = [randperm_L_all;randperm_L];
    end
    for iter_glob = 1:T_glob
        % W_local_all=cell(K,8);
        b_glob(:,iter_glob) = W_G{1,8};
        
        W_local_all=[];
        [A_t,W_t,sigma_all] = small_eval_aux_data_US_Census(W_G,X_aux_all,Y_aux_all,epochs,0.01,C);% for the aux-gradient method
        %tic
        for ii=1:K
            % take data from  client ii-divide info into K different chunks
            if(ii>1)
                randperm_L = randperm_L_all(N_all(ii-1)+1:N_all(ii-1)+N_all(ii));
            else
                randperm_L = randperm_L_all(1:N_all(1));
            end
            XTrain_10 = XTrain(randperm_L,:);
            YTrain_10 = YTrain(randperm_L);
            % flag_plot=1;
            [W_local,net] = small_NN_train_local_US_Census(XTrain_10,YTrain_10,epochs,B,W_G,flag_plot,C);
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
            %comparison:
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
        %toc
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
end

save('all_FL_comparison_new_aux_epochsE5_B256_US_Census_change_comparison_aux.mat','p_true','p_est_org','p_est_aux_grad','p_est_soft_label','p_est_compare','p_est_compare_mod','p_est_aux_W_grad','epochs','B','T_glob','b_updates','b_glob');

%%
updates_all = zeros(K,T_glob);

for t=1:T_glob
    for k=1:K
        updates_all(k,t) = norm(b_updates(:,k,t) - b_glob(:,t),'fro'); 
    end
end

%
figure;
plot((updates_all)','-o','color','blue'); 