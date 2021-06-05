function marker_index = select_dscore_markers(train_data)
global param;

train_methyaltion_data = train_data(1:(end-1),:);
num_all_markers = size(train_methyaltion_data,1);

if param.top_k >= num_all_markers
    marker_index = 1:num_all_markers;
    return;
end

train_labels = train_data(end,:);
train_class_index = unique(train_labels);
num_cancer_type = size(train_class_index,2)-1;

%% train healthy data
train_healthy_data = train_methyaltion_data(:,train_labels==1);
train_healthy_mean = mean(train_healthy_data,2);
train_healthy_std = std(train_healthy_data,[],2);

%%  train cancer data
train_cancer_mean = zeros(num_all_markers,num_cancer_type);
train_cancer_std = zeros(num_all_markers,num_cancer_type);
for i = 1:num_cancer_type
    class_index = train_labels==train_class_index(i+1);
    train_cancer_mean(:,i) = mean(train_methyaltion_data(:,class_index),2);
    train_cancer_std(:,i) = std(train_methyaltion_data(:,class_index),[],2);
end

%% max std
max_std = max([train_healthy_std,train_cancer_std],[],2);

%% min(d_hc)
min_d_hc = min(abs(repmat(train_healthy_mean,1,num_cancer_type) - train_cancer_mean),[],2);

%% min(d_cc)
all_d_cc = zeros(num_all_markers, num_cancer_type*(num_cancer_type-1)/2);
count = 1;
for m = 1:num_cancer_type-1
    for n = (m+1):num_cancer_type
        all_d_cc(:, count) = abs(train_cancer_mean(:,m)-train_cancer_mean(:,n));
        count = count + 1;
    end
end
min_d_cc = min(all_d_cc,[],2);

%% sort d-score
d_score = min_d_hc.*min_d_cc./max_std;
[~,sorted_index] = sort(-1*d_score);

%% marker index
marker_index = sorted_index(1:param.top_k)';

end