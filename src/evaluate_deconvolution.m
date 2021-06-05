function [MAE,RMSE,PCC]=evaluate_deconvolution(sRFDBI_results, test_theta)
% MAE is mean absolute error
% RMSE is root-mean-square error
% PCC is Pearson correlation coefficient
MAE = sum(abs(sRFDBI_results(2,:)- test_theta))/size(sRFDBI_results,2);
RMSE = sqrt(sum((sRFDBI_results(2,:) - test_theta).^2)/size(sRFDBI_results,2));
PCC = corr(sRFDBI_results(2,:)', test_theta');
end