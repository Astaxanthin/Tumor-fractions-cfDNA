clc;
clear;

%% initialization
dataset_name = 'real_dataset';  % 'simulation_dataset' and 'real_dataset'
Initialization;  % data loading, parameter setting, etc.

%% sREDBI algorithm
%-------------------------------------------------------------------%
%----Input: 
%------training data matrix, test data matrix, parameters
%----Output: 
%------diagnostic results: 
%------The first row represents the predicted class clabels
%------The second row represents the predicted tumor fraction
sRFDBI_results = sRFDBI(train_data, test_data);
%-------------------------------------------------------------------%

%% save results
save_name = strcat(save_dir,'sRFDBI_results.mat');
save(save_name, 'sRFDBI_results');

%% evaluate deconvolution (simulation dataset only)
if evaluate_deconvolution_flag
    [test_MAE,  test_RMSE, test_PCC] = evaluate_deconvolution(sRFDBI_results, test_theta)
end

%% evaluate diagnosis
confusion_matrix = compute_confusion_matrix(sRFDBI_results(1,:),param.test_sample_num)
ACC = diag(confusion_matrix)'./param.test_sample_num;
average_ACC = mean(ACC)
