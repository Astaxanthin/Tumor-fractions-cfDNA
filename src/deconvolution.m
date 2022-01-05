% Deconvolution experiments on simulation dataset
% Input: simulation dataset (training and test)
% Output: fraction vector for each sample
% Save: reference database

clear;clc;

%% parameter initialization

seed = 666;
rand('seed',seed);

data_root = '../data/';
param.p = 0.5;
param.mu = 1000;
param.class_num = 6;
param.cancer_pattern_num = 2;
param.healthy_pattern_num = 7;
param.test_sample_num = 20;
param.gradient = 10;
param.max_iterations = 1000;
param.supervised = 'false';
param.draw_graph = 'true';
param.weighted = false;
param.marker_num = inf;
param.generate_sample_num = 1000;


%% deconvolution on simulation dataset
dataset_name = 'simulation_dataset';
train_file_name = strcat(dataset_name,'/cnv_0/');
load(strcat(data_root,train_file_name,'train_data_part_1.mat'));
load(strcat(data_root,train_file_name,'train_data_part_2.mat'));
train_data = [train_data_part_1,train_data_part_2];
load(strcat(data_root,train_file_name,'train_theta.mat'));
train_save_add = strcat(data_root,train_file_name,'results/');
mkdir(train_save_add);

param.train_sample_num = repmat(200,1,6);

[reference_U,train_V,err, W_value] = NMF_train(train_data, nan,nan,param); % semi-reference-free deconvolution

save_name = strcat(train_save_add,'reference_U.mat');
save(save_name, 'reference_U');

%% deconvolution on test simulation data
cnv = 0.3; % choose a cnv event

test_file_name = strcat(dataset_name,'/cnv_',num2str(cnv),'/');
load(strcat(data_root,test_file_name,'test_data_part_1.mat'));
load(strcat(data_root,test_file_name,'test_data_part_2.mat'));
test_data = [test_data_part_1,test_data_part_2];
load(strcat(data_root,test_file_name,'test_theta.mat'));
test_save_add = strcat(data_root,test_file_name, 'Our_results/');
mkdir(test_save_add);

load(strcat(data_root,train_file_name, 'results/reference_U.mat'));
param.pseudo_label = zeros(1,size(test_data,2));
[test_V, ~] =  nnls_test(test_data, reference_U, param);

save_name = strcat(test_save_add,'test_V.mat');
save(save_name, 'test_V');

deconv_eval(test_V, test_theta, param);


%% deconvolution on real dataset
dataset_name = 'real_data';

load(strcat(data_root,dataset_name,'/validation_real_data.mat')); % load real data
load(strcat(data_root,train_file_name, 'results/reference_U.mat')); % load reference database learned from simulation dataset

param.pseudo_label = zeros(1,size(validation_real_data,2));
[validate_V, ~] =  nnls_test(validation_real_data, reference_U, param);

validate_save_add = strcat(data_root, 'real_data/results/');
mkdir(validate_save_add);

save_name = strcat(validate_save_add,'validate_V.mat');
save(save_name, 'validate_V');


%% deconvolution on Xu_data
file_add = strcat(data_root, dataset_name, '/Xu_data/');

load(strcat(file_add,'train_data.mat'));
load(strcat(file_add,'train_stage.mat'));
load(strcat(file_add,'test_data.mat'));
load(strcat(file_add,'test_stage.mat'));

result_save_add = strcat(file_add,'deconv_result/');
mkdir(result_save_add);
param.class_num = 2;
param.train_sample_num = [sum(train_stage==0), sum(train_stage>0)];

[reference_U,train_V,err, W_value] = NMF_train(train_data, nan,nan,param); % semi-reference-free deconvolution

save_name = strcat(result_save_add,'reference_U.mat');
save(save_name, 'reference_U');

param.pseudo_label = zeros(1,10000);
[train_V_without_label, ~] =  nnls_test(train_data, reference_U, param);
[test_V_without_label, ~] =  nnls_test(test_data, reference_U, param);

save_name = strcat(result_save_add,'train_V_without_label.mat');
save(save_name, 'train_V_without_label');
save_name = strcat(result_save_add,'test_V_without_label.mat');
save(save_name, 'test_V_without_label');


function deconv_eval(test_V,test_theta,param)

%% healthy evaluation
disp('>>>>>>>>Healthy evaluation<<<<<<<<<<');
[MAE,RMSE] = healthy_eval(test_V,test_theta,param)

%% tumor fraction evaluation
disp('>>>>>>>>Tumor fraction evaluation<<<<<<<<<<');
[MAE,RMSE,PCC] = tumor_eval(test_V,test_theta,param)
%
%% source fraction evaluation
disp('>>>>>>>>Source fraction evaluation<<<<<<<<<<');
[MAE,RMSE,PCC] = source_eval(test_V,test_theta,param)
disp('>>>>>>>>>>>>>End<<<<<<<<<<<<<<<');
disp('>>>>>>>>>>>>>----------------<<<<<<<<<<<<<<<');
end

function [MAE,RMSE] = healthy_eval(test_V,test_theta,param)
%% evaluate healthy samples
healthy_theta_pre = 1 - sum(test_V(1:param.healthy_pattern_num,1:param.test_sample_num*param.gradient),1);
healthy_theta_gt =1 -  test_theta(1:param.test_sample_num*param.gradient);
MAE = sum(abs(healthy_theta_pre - healthy_theta_gt))/size(healthy_theta_pre,2);
RMSE = sqrt(sum((healthy_theta_pre - healthy_theta_gt).^2)/size(healthy_theta_pre,2));
end

function [MAE,RMSE,PCC] = tumor_eval(test_V,test_theta,param)
%% evaluate tumor samples
predict_theta = 1-sum(test_V(1:param.healthy_pattern_num,:),1);
[MAE,RMSE,PCC]=evaluate_deconvolution(predict_theta, test_theta,param.test_sample_num*param.gradient);
end

function [MAE,RMSE,PCC] = source_eval(test_V,test_theta,param)
%% evaluate source fraction prediction
h_mask = [ones(1,param.healthy_pattern_num),zeros(1,param.cancer_pattern_num*(param.class_num-1))];
cancer_mask = [zeros(param.class_num-1,param.healthy_pattern_num),kron(eye(param.class_num-1,param.class_num-1),ones(1,param.cancer_pattern_num))];
mask = [h_mask;cancer_mask];
predict_theta_mat = mask*test_V;
predict_theta = [];
for i = 1:param.class_num
    predict_theta = [predict_theta, predict_theta_mat(i,(i-1)*param.test_sample_num*param.gradient+1:i*param.test_sample_num*param.gradient)];
end
%         predict_theta = 1-sum(train_V_without_label(1:param.healthy_pattern_num,:),1);
[MAE,RMSE,PCC]=evaluate_deconvolution(predict_theta, test_theta,param.test_sample_num*param.gradient);
end

function [MAE,RMSE,PCC]=evaluate_deconvolution(predict_theta, test_theta,healthy_sample_num)
% MAE is mean absolute error
% RMSE is root-mean-square error
% PCC is Pearson correlation coefficient

index_st = healthy_sample_num + 1;
predict_theta = predict_theta(index_st:end);
test_theta = test_theta(index_st:end);

MAE = sum(abs(predict_theta - test_theta))/size(predict_theta,2);
RMSE = sqrt(sum((predict_theta - test_theta).^2)/size(predict_theta,2));
PCC = corr(predict_theta', test_theta');
end