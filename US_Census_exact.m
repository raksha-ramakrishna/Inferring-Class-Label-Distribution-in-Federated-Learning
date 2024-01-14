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
    
    layers(1,1).Mean=[35.400002,0,0.10000000,0,0.69999999,0,0.10000000,0.10000000,0,137121.91,0,0,0,0,0,0,0.20000000,0,0.30000001,0.10000000,0.10000000,0,0.30000001,0,0,0,0.20000000,0,0.40000001,0,0.40000001,0,0,0,0,0.10000000,0,0,0,0,0.40000001,0,0.10000000,0,0.20000000,0.10000000,0.10000000,0.30000001,0.20000000,0.10000000,0.30000001,0.10000000,0,0,0.10000000,0,0,0.89999998,0.40000001,0.60000002,747.59998,0,41.500000,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0];
    layers(1,1).StandardDeviation = [12.281693,0,0.30000001,0,0.45825756,0,0.30000001,0.30000001,0,83490.750,0,0,0,0,0,0,0.40000001,0,0.45825756,0.30000001,0.30000001,0,0.45825756,0,0,0,0.40000001,0,0.48989794,0,0.48989794,0,0,0,0,0.30000001,0,0,0,0,0.48989794,0,0.30000001,0,0.40000001,0.30000001,0.30000001,0.45825756,0.40000001,0.30000001,0.45825756,0.30000001,0,0,0.30000001,0,0,0.30000001,0.48989794,0.48989794,1600.2338,0,9.5000000,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
    layers(2,1).Weights = unifrnd(-1,1,32,numFeatures);
    layers(2,1).Bias = unifrnd(-1,1,32,1);
    
    layers(3,1).Weights = unifrnd(-1,1,16,32);
    layers(3,1).Bias = unifrnd(-1,1,16,1);
    
    layers(4,1).Weights = unifrnd(-1,1,8,16);
    layers(4,1).Bias = unifrnd(-1,1,8,1);
    
   % layers(5,1).Weights = unifrnd(-1,1,2,8);
   layers(5,1).Weights = zeros(2,8); 
    layers(5,1).Bias = unifrnd(-1,1,2,1);
    
    W_G_0 = {layers(2,1).Weights,layers(2,1).Bias,layers(3,1).Weights,layers(3,1).Bias,...
        layers(4,1).Weights,layers(4,1).Bias,layers(5,1).Weights,layers(5,1).Bias};
    W_G = W_G_0;
    
    %% client selection:
    N_all = zeros(K,1); %per client
    LB = floor(length(YTrain)./K); %how much data can each client have.
    randperm_L_all = [];
    
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
        
        % this is to divide the data into many clients-change for every
        % iteration
        b_glob(:,iter_glob) = W_G{1,8};
        
        W_local_all=[];
        net1=assembleNetwork(layers);
        [A_t,W_t,sigma_all] = small_eval_aux_data_US_Census(W_G,X_aux_all,Y_aux_all,epochs,0.01,C,net1);% for the aux-gradient method
        %tic
        for ii=1:K
            % take data from  client ii-divide info into K different chunks
            if(ii>1)
                randperm_L = randperm_L_all(sum(N_all(1:ii-1))+1:sum(N_all(1:ii)));
            else
                randperm_L = randperm_L_all(1:N_all(1));
            end
            XTrain_10 = XTrain(randperm_L,:);
            YTrain_10 = YTrain(randperm_L);
            % flag_plot=1;
            [W_local,net] = small_NN_train_local_US_Census(XTrain_10,YTrain_10,epochs,N_all(ii),W_G,flag_plot,C);
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
    save('US_census_exact.mat','p_true','p_est_org','p_est_aux_grad','p_est_soft_label','p_est_compare','p_est_compare_mod','p_est_aux_W_grad','epochs','B','T_glob','b_updates','b_glob');

end
%%
ip_filename='US_census_exact.mat';
op_filename = 'US_Census_exact';
[mean_KL,mean_MSE] = KL_divergence_and_MSE(ip_filename,op_filename,2,10,1,10);

%%
load(ip_filename);
t_n = 10; %iteration of interest
C = 2;
p_us = zeros(C,200);
p_them = zeros(C,200);
p_actual = zeros(C,200);
K=10;
m=1; %index of our method to be chosen
for ff=1:20
    for k=1:10
        tmp1 = SimplexProj(transpose(p_est_aux_W_grad(:,k,t_n,ff)));
        p_us(:,(ff-1)*K+k) = tmp1';
        
        tmp2 = SimplexProj(transpose(p_est_compare_mod(:,k,t_n,ff)));
        p_them(:,(ff-1)*K+k) = tmp2';
        
        p_actual(:,(ff-1)*K+k) = p_true(:,k,t_n,ff);
        
    end

end
%%
figure;
plot(p_actual(1,:),p_actual(1,:),'*');
hold on
plot(p_actual(1,:),p_us(1,:),'.');
plot(p_actual(1,:),p_them(1,:),'x','color','black');
grid on
legend('actual','aux-weight grad','Wang et.al modified');
xlabel('p');
ylabel('$\hat{p}$');
%%
B = csvread(strcat('MSE_',op_filename,num2str(t_n-1),'.csv'));


%
m = 1; %index of our method to be chosen
c = 6; %their method
figure;
plot(p_all,B(:,m),'*');
hold on
plot(p_all,2*p_all.^2,'.');
plot(p_all,2*(1-p_all).^2,'.');
hold on
plot(p_all,B(:,c),'o');
%%
%load('new_comparison_US_Census_only_1_trial_iter.mat');
K=10;
fl=5;
C = 2;
%%
p_init = zeros(2,K,fl);
%p_true_val = reshape(p_true(:,:,1,:),C,K,fl-2);
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
for ff = 1:fl
    for t=1:T_glob
        for k=1:K
            
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
pp=50;
for tt = 1:T_glob
    R = [errKL_reg(1:pp,tt),errKL(1:pp,tt),errKL_aux(1:pp,tt),errKL_compare(1:pp,tt),errKL_compare_mod(1:pp,tt),errKL_aux_W(1:pp,tt)];
    csvwrite(strcat('KL_USCensus',num2str(tt-1),'.csv'),R);
end


%%
R = [errKL_reg(:),errKL(:),errKL_aux(:),errKL_compare(:),errKL_compare_mod(:),errKL_aux_W(:)];
figure;
boxplot(R);
%csvwrite(strcat('KL_reg_USCensus_E_5_aux_new','.csv'),R);
figure;
plot(mean(R),'-o')

csvwrite(strcat('KL_reg_USCensus_E_5_aux_new_only1','.csv'),R);
%%
% KL divergence curve 2-d
p1 = 0:0.01:1; q1 = 1-p1;
p2 = 0:0.01:1; q2 = 1-p2;

f1 = [p1;q1]; f2 = [p2;q2];
KLDiv=zeros(length(p1),length(p2));
MSE = zeros(length(p1),length(p2));

for i=1:length(p1)
    for j=1:length(p2)
        KLDiv(i,j) = sum(f1(:,i) .* log((f1(:,i)+eps)./(f2(:,j)+eps)));
        MSE(i,j) = meansqr(f1(:,i)-f2(:,j));
    end
end

figure;
surf(p1,p2,log(KLDiv))

figure;
surf(p1,p2,log(MSE));
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