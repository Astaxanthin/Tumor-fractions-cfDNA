clear;clc;

%% Parameter initialization

seed = 666;
rand('seed',seed);

data_root = '../data/';
dataset_name = 'simulation_data';
param.p = 0.5;
param.mu = 1000;
param.class_num = 6;
param.test_sample_num = 20;
param.gradient = 10;
param.max_iterations = 1000;
param.supervised = 'false';
param.draw_graph = 'true';
param.weighted = false;
param.marker_num = inf;
param.generate_sample_num = 1000;

%% Diagnosis on simulation dataset

train_file_name = strcat(dataset_name,'/cnv_0/'); 
load(strcat(data_root,train_file_name,'train_data.mat')); % load training data
load(strcat(data_root,train_file_name,'train_theta.mat'));

cnv = 0.3;
test_file_name = strcat(dataset_name,'/cnv_',num2str(cnv),'/');
load(strcat(data_root,test_file_name,'test_data.mat')); % load test data
load(strcat(data_root,test_file_name,'test_theta.mat'));

param.cancer_pattern_num = 2;
param.healthy_pattern_num = 7;

burden_thd = 0.1;
num_control = 200;
num_per_cancer = 40;

seed = randperm(1000);

%    >>>>>>>>>>sampling training samples from all stages<<<<<<<<<
for i = 1:10
    rand('seed',seed(i));
    subfile_dir = strcat('train_all_h200_c40/subset_',num2str(i),'/');
    %     subfile_dir = 'train_all_500/';
    tic;
    
    sample_index = randperm(num_control);
    sample_index = sample_index(1:num_control);
    for i = 2:param.class_num
        rand_index = randperm(num_control);
        sample_index = [sample_index, (i-1)*num_control + rand_index(1:num_per_cancer)];
    end
    train_data_all = train_data(:,sample_index);
    train_theta_all = train_theta(:,sample_index);
    
    train_data_sample = train_data_all;
    
    result_save_add = strcat(data_root, test_file_name,'diagnosis_results/',subfile_dir);
    mkdir(result_save_add);
    
    save_name = strcat(result_save_add,'/train_data_sample.mat');
    save(save_name, 'train_data_sample');
    
    param.weighted = false;
    param.train_sample_num = [num_control, repmat(num_per_cancer,1,5)];
    
    [reference_U,train_V,err, W_value] = NMF_train(train_data_sample, nan,nan,param);
    
    save_name = strcat(result_save_add,'reference_U.mat');
    save(save_name, 'reference_U');
    
    early_index = [1:200, find(test_theta<0.1)];
    test_data_early = test_data(:, early_index);
    
    save_name = strcat(result_save_add,'/test_data_early.mat');
    save(save_name, 'test_data_early');
    
    %     >>>>>>>>>>>>>>>adopt original data<<<<<<<<<<<<<<<
    load(strcat(data_root, test_file_name, 'diagnosis_results/',subfile_dir,'/train_data_sample.mat'));
    load(strcat(data_root, test_file_name, 'diagnosis_results/',subfile_dir,'/reference_U.mat'));
    param.pseudo_label = zeros(1,10000);
    
    [train_V_without_label, ~] =  nnls_test(train_data_sample, reference_U, param);
    [test_early_V_without_label, ~] =  nnls_test(test_data_early, reference_U, param);
    [test_all_V_without_label, ~] =  nnls_test(test_data, reference_U, param);
    
    save_name = strcat(result_save_add,'/train_V_without_label.mat');
    save(save_name, 'train_V_without_label');
    
    save_name = strcat(result_save_add,'/test_early_V_without_label.mat');
    save(save_name, 'test_early_V_without_label');
    
    save_name = strcat(result_save_add,'/test_all_V_without_label.mat');
    save(save_name, 'test_all_V_without_label');
    
    toc;
end

% >>>>>>>>>>sampling training samples from late stages<<<<<<<<<
for i = 1:10
    rand('seed',seed(i));
    subfile_dir = strcat('train_late_h200_c40/subset_',num2str(i),'/');
    %     subfile_dir = 'train_all_500/';
    tic;
    
    sample_index = randperm(num_control);
    sample_index = sample_index(1:num_control);
    for i = 2:param.class_num
        late_cancer_index = find(train_theta>burden_thd&train_theta<1);
        late_cancer_index(late_cancer_index<=(i-1)*num_control | late_cancer_index>i*200) = [];
        rand_index = randperm(length(late_cancer_index));
        sample_index = [sample_index, late_cancer_index(rand_index(1:num_per_cancer))];
    end
    train_data_late = train_data(:,sample_index);
    train_theta_late = train_theta(:,sample_index);
    
    result_save_add = strcat(data_root,test_file_name,'diagnosis_results/',subfile_dir);
    mkdir(result_save_add);
    
    save_name = strcat(result_save_add, '/train_data_late.mat');
    save(save_name, 'train_data_late');
    
    param.train_sample_num = [num_control, repmat(num_per_cancer,1,5)];
    
    [reference_U,train_V,err, W_value] = NMF_train(train_data_late, nan,nan,param);
    
    save_name = strcat(result_save_add,'reference_U.mat');
    save(save_name, 'reference_U');
    
    early_index = [1:200, find(test_theta<0.1)];
    test_data_early = test_data(:, early_index);
    
    save_name = strcat(result_save_add,'/test_data_early.mat');
    save(save_name, 'test_data_early');
    
    %     >>>>>>>>>>>>>>>adopt original data<<<<<<<<<<<<<<<
    load(strcat(data_root,test_file_name, 'diagnosis_results/',subfile_dir,'/train_data_late.mat'));
    load(strcat(data_root,test_file_name, 'diagnosis_results/',subfile_dir,'/reference_U.mat'));
    param.pseudo_label = zeros(1,10000);
    
    [train_V_without_label, ~] =  nnls_test(train_data_late, reference_U, param);
    [test_early_V_without_label, ~] =  nnls_test(test_data_early, reference_U, param);
    [test_all_V_without_label, ~] =  nnls_test(test_data, reference_U, param);
    
    save_name = strcat(result_save_add,'/train_V_without_label.mat');
    save(save_name, 'train_V_without_label');
    
    save_name = strcat(result_save_add,'/test_early_V_without_label.mat');
    save(save_name, 'test_early_V_without_label');
    
    save_name = strcat(result_save_add,'/test_all_V_without_label.mat');
    save(save_name, 'test_all_V_without_label');
    
    toc;
end

%% all stages for training
ALL_acc_all = zeros(16,10);
for sub_id = 1:10
    subfile_dir = strcat('train_all_h200_c40/subset_',num2str(sub_id),'/');
    result_save_add = strcat(data_root,test_file_name, 'diagnosis_results/',subfile_dir);
    load(strcat(result_save_add,'/train_data_sample.mat'));
    load(strcat(result_save_add,'/train_V_without_label.mat'));
    
    train_gt_label = [ones(1,200),kron(2:6,ones(1,40))];
    
    load(strcat(result_save_add,'/test_data_early.mat'));
    load(strcat(result_save_add,'/test_early_V_without_label.mat'));
    
    load(strcat(data_root,test_file_name,'test_data.mat'));
    load(strcat(result_save_add,'/test_all_V_without_label.mat'));
    
    test_early_gt_label = [ones(1,200),kron(2:6,ones(1,40))];
    test_early_sample_num = [200,repmat(40,1,5)];
    
    test_all_gt_label = [ones(1,200),kron(2:6,ones(1,200))];
    test_all_sample_num = [200,repmat(200,1,5)];
    
    method = {'RF','SVM','MLP','Naive'};
    for i = 1:2:7
        pred_prob_all = zeros(param.class_num, size(test_data_early,2),10);
        bayes_pred_prob_all = zeros(param.class_num, size(test_data_early,2),10);
        for j = 1:10
            [pred_prob, pred_labels, pred_bayes_prob, pred_bayes_labels] = BayesDiagnosis(train_data_sample, train_gt_label,...
                test_data_early, method{round(i/2)}, train_V_without_label, test_early_V_without_label, param);
            pred_prob_all(:,:,j) = pred_prob';
            bayes_pred_prob_all(:,:,j) = pred_bayes_prob;
        end
        
        mean_pre_prob = mean(pred_prob_all,3);
        mean_bayes_pre_prob = mean(bayes_pred_prob_all,3);
        
        save_name = strcat(result_save_add,'/early/',method{round(i/2)},'_mean_pre_prob.mat');
        save(save_name, 'mean_pre_prob');
        
        save_name = strcat(result_save_add,'/early/', method{round(i/2)},'_mean_bayes_pre_prob.mat');
        save(save_name, 'mean_bayes_pre_prob');
        
        [~, final_pred_labels] = max(mean_pre_prob,[],1);
        [~, final_bayes_pred_labels] = max(mean_bayes_pre_prob,[],1);
        
        confusion_matrix = confusionmat(final_pred_labels',test_early_gt_label)'
        precision = mean([diag(confusion_matrix)'./test_early_sample_num])
        
        bayes_confusion_matrix = confusionmat(final_bayes_pred_labels',test_early_gt_label)'
        bayes_precision = mean([diag(bayes_confusion_matrix)'./test_early_sample_num])
        
        ALL_acc_all(i, sub_id) = precision;
        ALL_acc_all(i+1, sub_id) = bayes_precision;
    end
    
    for k = 1:2:7
        pred_prob_all = zeros(param.class_num, size(test_data,2),10);
        bayes_pred_prob_all = zeros(param.class_num, size(test_data,2),10);
        for j = 1:10
            [pred_prob, pred_labels, pred_bayes_prob, pred_bayes_labels] = BayesDiagnosis(train_data_sample, train_gt_label,...
                test_data, method{round(k/2)}, train_V_without_label, test_all_V_without_label, param);
            pred_prob_all(:,:,j) = pred_prob';
            bayes_pred_prob_all(:,:,j) = pred_bayes_prob;
        end
        
        mean_pre_prob = mean(pred_prob_all,3);
        mean_bayes_pre_prob = mean(bayes_pred_prob_all,3);
        
        save_name = strcat(result_save_add,'/all/',method{round(k/2)},'_mean_pre_prob.mat');
        save(save_name, 'mean_pre_prob');
        
        save_name = strcat(result_save_add,'/all/', method{round(k/2)},'_mean_bayes_pre_prob.mat');
        save(save_name, 'mean_bayes_pre_prob');
        
        [~, final_pred_labels] = max(mean_pre_prob,[],1);
        [~, final_bayes_pred_labels] = max(mean_bayes_pre_prob,[],1);
        
        confusion_matrix = confusionmat(final_pred_labels',test_all_gt_label)'
        precision = mean([diag(confusion_matrix)'./test_all_sample_num])
        
        bayes_confusion_matrix = confusionmat(final_bayes_pred_labels',test_all_gt_label)'
        bayes_precision = mean([diag(bayes_confusion_matrix)'./test_all_sample_num])
        
        ALL_acc_all(k+8, sub_id) = precision;
        ALL_acc_all(k+9, sub_id) = bayes_precision;
    end
end


%% late-stage for training 
ALL_acc_late = zeros(16,10);
for sub_id = 1:10
    subfile_dir = strcat('train_late_h200_c40/subset_',num2str(sub_id),'/');
    result_save_add = strcat(data_root,test_file_name, 'diagnosis_results/',subfile_dir);
    load(strcat(result_save_add,'/train_data_late.mat'));
    load(strcat(result_save_add,'/train_V_without_label.mat'));
    
    train_gt_label = [ones(1,200),kron(2:6,ones(1,40))];
    
    load(strcat(result_save_add,'/test_data_early.mat'));
    load(strcat(result_save_add,'/test_early_V_without_label.mat'));
    
    load(strcat(data_root,test_file_name,'test_data.mat'));
    load(strcat(result_save_add,'/test_all_V_without_label.mat'));
    
    test_early_gt_label = [ones(1,200),kron(2:6,ones(1,40))];
    test_early_sample_num = [200,repmat(40,1,5)];
    
    test_all_gt_label = [ones(1,200),kron(2:6,ones(1,200))];
    test_all_sample_num = [200,repmat(200,1,5)];
    
    method = {'RF','SVM','MLP','Naive'};
    for i = 1:2:7
        pred_prob_all = zeros(param.class_num, size(test_data_early,2),10);
        bayes_pred_prob_all = zeros(param.class_num, size(test_data_early,2),10);
        for j = 1:10
            [pred_prob, pred_labels, pred_bayes_prob, pred_bayes_labels] = BayesDiagnosis(train_data_late, train_gt_label,...
                test_data_early, method{round(i/2)}, train_V_without_label, test_early_V_without_label, param);
            pred_prob_all(:,:,j) = pred_prob';
            bayes_pred_prob_all(:,:,j) = pred_bayes_prob;
        end
        
        mean_pre_prob = mean(pred_prob_all,3);
        mean_bayes_pre_prob = mean(bayes_pred_prob_all,3);
        
        save_name = strcat(result_save_add,'/early/',method{round(i/2)},'_mean_pre_prob.mat');
        save(save_name, 'mean_pre_prob');
        
        save_name = strcat(result_save_add,'/early/', method{round(i/2)},'_mean_bayes_pre_prob.mat');
        save(save_name, 'mean_bayes_pre_prob');
        
        [~, final_pred_labels] = max(mean_pre_prob,[],1);
        [~, final_bayes_pred_labels] = max(mean_bayes_pre_prob,[],1);
        
        confusion_matrix = confusionmat(final_pred_labels',test_early_gt_label)'
        precision = mean([diag(confusion_matrix)'./test_early_sample_num])
        
        bayes_confusion_matrix = confusionmat(final_bayes_pred_labels',test_early_gt_label)'
        bayes_precision = mean([diag(bayes_confusion_matrix)'./test_early_sample_num])
        
        ALL_acc_late(i,sub_id) = precision;
        ALL_acc_late(i+1,sub_id) = bayes_precision;
    end
    
    for k = 1:2:7
        pred_prob_all = zeros(param.class_num, size(test_data,2),10);
        bayes_pred_prob_all = zeros(param.class_num, size(test_data,2),10);
        for j = 1:10
            [pred_prob, pred_labels, pred_bayes_prob, pred_bayes_labels] = BayesDiagnosis(train_data_late, train_gt_label,...
                test_data, method{round(k/2)}, train_V_without_label, test_all_V_without_label, param);
            pred_prob_all(:,:,j) = pred_prob';
            bayes_pred_prob_all(:,:,j) = pred_bayes_prob;
        end
        
        mean_pre_prob = mean(pred_prob_all,3);
        mean_bayes_pre_prob = mean(bayes_pred_prob_all,3);
        
        save_name = strcat(result_save_add,'/all/',method{round(k/2)},'_mean_pre_prob.mat');
        save(save_name, 'mean_pre_prob');
        
        save_name = strcat(result_save_add,'/all/', method{round(k/2)},'_mean_bayes_pre_prob.mat');
        save(save_name, 'mean_bayes_pre_prob');
        
        [~, final_pred_labels] = max(mean_pre_prob,[],1);
        [~, final_bayes_pred_labels] = max(mean_bayes_pre_prob,[],1);
        
        confusion_matrix = confusionmat(final_pred_labels',test_all_gt_label)'
        precision = mean([diag(confusion_matrix)'./test_all_sample_num])
        
        bayes_confusion_matrix = confusionmat(final_bayes_pred_labels',test_all_gt_label)'
        bayes_precision = mean([diag(bayes_confusion_matrix)'./test_all_sample_num])
        
        ALL_acc_late(k+8,sub_id) = precision;
        ALL_acc_late(k+9,sub_id) = bayes_precision;
    end
end