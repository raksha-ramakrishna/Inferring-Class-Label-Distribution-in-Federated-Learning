load('US_census_address_class_imbalance_no_change_50_clients_200_iter_acc.mat');
load('data_for_US_census.mat');
iter_glob=200;
accuracy_no_change = accuracy(1:iter_glob);
figure;
plot(accuracy(1:iter_glob),'-o');
hold on
load('US_census_address_class_imbalance_remove_iter_5_change_only_p_less_clients_200_iter_acc.mat');
plot(accuracy(1:iter_glob),'-*');
grid on
legend('Without client removal','client removal using p^{init}_{bias} after 5 iterations');
xlabel('global iteration t');
ylabel('AUC');

csvwrite('AUC_scores.csv',[transpose(1:iter_glob),accuracy_no_change,accuracy(1:iter_glob)]);