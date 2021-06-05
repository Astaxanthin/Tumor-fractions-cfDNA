function [classifier_prediction, sRFDBI_results] = sRFDBI(train_data, test_data)

global param;

train_labels = train_data(end,:);
[train_labels,sorted_train_index] = sort(train_labels);
train_data = train_data(1:(end-1),sorted_train_index);

test_labels = test_data(end,:);
[test_labels,sorted_test_index] = sort(test_labels);
test_data = test_data(1:(end-1),sorted_test_index);

train_class_index = unique(train_labels);
param.class_num = size(train_class_index,2);
param.train_sample_num = [];
for i = 1:size(train_class_index,2)
    param.train_sample_num = [param.train_sample_num, sum(train_labels==train_class_index(i))];
end

test_class_index = unique(test_labels);
param.test_sample_num = [];
for i = 1:size(test_class_index,2)
    param.test_sample_num = [param.test_sample_num, sum(test_labels==test_class_index(i))];
end

%% structural reference-free deconvolution (sRFD)
fprintf('1. Performing structural reference-free deconvolution (sRFD)\n');
fprintf('This may take a while...\n');
[train_W,train_H,err] = sRFD(train_data,param);

%% coarse tissue fraction (deconcolution without structural constraints)
fprintf('2. Estimating coarse tissue fractions\n');
coarse_train_H =  deconvolution_with_reference(train_data, train_W, zeros(1,size(train_data,2)), param);
coarse_test_H =  deconvolution_with_reference(test_data, train_W, zeros(1,size(test_data,2)), param);

%% Bayesian inference (BI)
fprintf('3. Performing Bayesian inference\n');
[classifier_prob, classifier_prediction, sRFDBI_prob, sRFDBI_prediction] = BI(train_data,test_data, param.prior, coarse_train_H, coarse_test_H, param);

%% refined tissue fraction
fprintf('4. Refining tissue fraction prediction\n');
refined_test_H = deconvolution_with_reference(test_data, train_W, sRFDBI_prediction, param);

%% final results
% Two rows: the first row is predicted labels, the second row is predicted tumor fraction.
refined_tumor_fraction = sum(refined_test_H((param.healthy_pattern_num+1):end,:),1);
sRFDBI_results = [sRFDBI_prediction;refined_tumor_fraction];
fprintf('Finish!\n');

end
