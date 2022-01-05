clear;clc;

seed = 888; 
rand('seed',seed);

file_add = '../data/real_dataset/Chen_data/';
cancer_liquid_type = {'colon','liver','esophagus','lung','stomach'};
cancer_tissue_type = {'colon','lung','stomach'};

%% Parameter initialization 
param.p = 0.5;
param.mu = 1000;
param.class_num = 6;
param.max_iterations = 1000;
param.supervised = 'false';
param.draw_graph = 'true';
param.weighted = false;
param.marker_num = inf;
param.generate_sample_num = 1000;
param.cancer_pattern_num = 2;
param.healthy_pattern_num = 7;

%% late-stage patients for model training
data_add = strcat(file_add, 'late_stage_training/');
load(strcat(data_add,'train_data.mat'));

result_save_add = strcat(data_add,'result/');
mkdir(result_save_add);

param.train_sample_num = [207,5,20,39,45,35];
param.test_sample_num = [207,2,3,29,11,34];

train_gt_label = [];
for i = 1:size(param.train_sample_num,2)
    train_gt_label = [train_gt_label, i*ones(1,param.train_sample_num(i))];
end
test_gt_label = [];
for i = 1:size(param.test_sample_num,2)
    test_gt_label = [test_gt_label, i*ones(1,param.test_sample_num(i))];
end

% semi-reference-free deconvolution
[reference_U,train_V,err, W_value] = NMF_train(train_data, nan,nan,param);

save_name = strcat(result_save_add,'reference_U.mat');
save(save_name, 'reference_U');

% extend data using borderline-SMOTE
[extend_V,extend_labels] = borderline_smote(train_V, train_gt_label, 40,'fraction', param);
reconstruction_err = train_data-reference_U*train_V;
extend_data = mix_WH(reference_U, extend_V, extend_labels,reconstruction_err,param);

enh_train_data = [train_data, extend_data];
enh_train_gt_label = [train_gt_label, extend_labels];
[enh_train_gt_label,sorted_index] = sort(enh_train_gt_label);
enh_train_data = enh_train_data(:,sorted_index);

save_name = strcat(result_save_add, 'enh_train_data.mat');
save(save_name, 'enh_train_data');
save_name = strcat(result_save_add, 'enh_train_gt_label.mat');
save(save_name, 'enh_train_gt_label');

train_class_index = unique(enh_train_gt_label);
param.extended_train_sample_num = [];
for i = 1:size(train_class_index,2)
    param.extended_train_sample_num = [param.extended_train_sample_num, sum(enh_train_gt_label==train_class_index(i))];
end

param.train_sample_num = param.extended_train_sample_num;

% deconvolution on test samples (healthy individuals and early-stage patients)
load(strcat(data_add,'test_data.mat'));

load(strcat(result_save_add,'/enh_train_data.mat'));
load(strcat(result_save_add,'/reference_U.mat'));

param.pseudo_label = zeros(1,10000);
[enh_train_V_without_label, ~] =  nnls_test(enh_train_data, reference_U, param);
[train_V_without_label, ~] =  nnls_test(train_data, reference_U, param);
[test_V_without_label, ~] =  nnls_test(test_data, reference_U, param);

save_name = strcat(result_save_add,'train_V_without_label.mat');
save(save_name, 'train_V_without_label');
save_name = strcat(result_save_add,'enh_train_V_without_label.mat');
save(save_name, 'enh_train_V_without_label');
save_name = strcat(result_save_add,'test_V_without_label.mat');
save(save_name, 'test_V_without_label');

% deconvolution on pre-diagnosis samples
load(strcat(result_save_add,'/reference_U.mat'));
load(strcat(data_add,'pre_liquid_data.mat'));
param.pseudo_label = zeros(1,10000);
[validate_V_without_label, ~] =  nnls_test(pre_liquid_data, reference_U, param);
save_name = strcat(result_save_add,'validate_V_without_label.mat');
save(save_name, 'validate_V_without_label');

%% Diagnosis
ALL_acc_late = zeros(16,1);
tic;
data_add = strcat(file_add, 'late_stage_training/');
result_save_add = strcat(data_add,'result/');
load(strcat(data_add,'train_data.mat'));
load(strcat(data_add,'test_data.mat'));
load(strcat(result_save_add,'enh_train_data.mat'));
load(strcat(result_save_add,'enh_train_gt_label.mat'));
load(strcat(result_save_add,'train_V_without_label.mat'));
load(strcat(result_save_add,'enh_train_V_without_label.mat'));
load(strcat(result_save_add,'test_V_without_label.mat'));

train_gt_label = [];
for i = 1:size(param.train_sample_num,2)
    train_gt_label = [train_gt_label, i*ones(1,param.train_sample_num(i))];
end
test_gt_label = [];
for i = 1:size(param.test_sample_num,2)
    test_gt_label = [test_gt_label, i*ones(1,param.test_sample_num(i))];
end

method = {'RF','SVM','MLP'};

select_method = method{2};

% use enhanced data after oversampling
pred_prob_all = zeros(param.class_num, size(test_data,2),10);
bayes_pred_prob_all = zeros(param.class_num, size(test_data,2),10);
model_save_add = strcat(result_save_add,'/enh/',select_method,'/');
mkdir(model_save_add);
for j = 1:10
    [pred_prob, pred_labels, pred_bayes_prob, pred_bayes_labels, save_model] = BayesDiagnosis(enh_train_data, enh_train_gt_label,...
        test_data, select_method, enh_train_V_without_label, test_V_without_label, param, j);
    pred_prob_all(:,:,j) = pred_prob';
    bayes_pred_prob_all(:,:,j) = pred_bayes_prob;
    
    save(strcat(model_save_add, select_method, 'model_', num2str(j)), 'save_model');
end

mean_pre_prob = mean(pred_prob_all,3);
mean_bayes_pre_prob = mean(bayes_pred_prob_all,3);

save_name = strcat(result_save_add,'/enh/',select_method,'_mean_pre_prob.mat');
save(save_name, 'mean_pre_prob');

save_name = strcat(result_save_add,'/enh/', select_method,'_mean_bayes_pre_prob.mat');
save(save_name, 'mean_bayes_pre_prob');

[~, final_pred_labels] = max(mean_pre_prob,[],1);
[~, final_bayes_pred_labels] = max(mean_bayes_pre_prob,[],1);

confusion_matrix = confusionmat(final_pred_labels',test_gt_label)'
precision = mean([diag(confusion_matrix)'./param.test_sample_num])

bayes_confusion_matrix = confusionmat(final_bayes_pred_labels',test_gt_label)'
bayes_precision = mean([diag(bayes_confusion_matrix)'./param.test_sample_num])

test = 1;