function [U,V,err,weight_val]=NMF_train(train_X, refer_U, refer_V, param)

seed = 666;
rand('seed',seed);

class_num=param.class_num;
healthy_pattern_num = param.healthy_pattern_num;
cancer_pattern_num = param.cancer_pattern_num;
r=param.healthy_pattern_num + (class_num-1)*cancer_pattern_num; %
train_sample_num = param.train_sample_num;

%% calculate weight matrix among cancer types
% weight_all = zeros(1,size(train_X,1));
% edges = 0:0.1:1;
% for i = 1:size(train_X,1)
%     each_marker_data = train_X(i,:);
%     train_sample_index_st = cumsum(train_sample_num);
%     hist_all = zeros(10,size(train_sample_num,2)-1);
%     for j = 1:size(train_sample_index_st,2)-1
%         cancer_type_data = each_marker_data(train_sample_index_st(j)+1:train_sample_index_st(j+1));
%         hist_all(:,j) = histcounts(cancer_type_data,edges)/size(cancer_type_data,2);
%     end
%     d = svd(hist_all);
%     om = 1/(sqrt(size(train_sample_index_st,2)-1)-1)*(sum(d)/norm(hist_all,'fro')-1);
%     weight_all(i) = om;
% end
% 
% [sorted_weight, sort_index] = sort(weight_all, 'descend');
% if param.marker_num ~= inf
%     weight_val = sorted_weight(1:param.marker_num);
%     selected_index = sort_index(1:param.marker_num);
% else
%     weight_val = weight_all;
%     selected_index = 1:size(train_X,1);
% end

%% normalize weight
% weight_val = exp((weight_val-mean(weight_val))/(std(weight_val)));

if ~param.weighted
    weight_val = ones(1,size(train_X,1));
else
    weight_val = param.marker_weight';
%     selected_index = 1:size(train_X,1);
    weight_val(weight_val==0) = 1e-5;
end

%% deconvolution
m=size(train_X,1);
n1=size(train_X,2);
% n2 = size(test_X,2);

% U=rand([m,r]); %
% V=rand([r,size(rand_index,2)]);
% norm_coff=sum(V,1);
% V=V./repmat(norm_coff,size(V,1),1);%normalization

if isnan(refer_U)
    U=rand([m,r]); %
else
    U = refer_U;
end
if isnan(refer_V)
    V=rand([r,n1]);
    norm_coff=sum(V,1);
    V=V./repmat(norm_coff,size(V,1),1);%normalization
else
    V = refer_V;
end

%% RSNMF 2017 TNN
healthy_mask = [zeros(healthy_pattern_num, train_sample_num(1));ones(r-healthy_pattern_num, train_sample_num(1))];
cancer_tissue_mask_top = zeros(healthy_pattern_num, sum(train_sample_num)-train_sample_num(1));

cancer_tissue_mask_down = [];
for i = 2:size(train_sample_num,2)
    cancer_tissue_mask_down =  blkdiag(cancer_tissue_mask_down, ones(cancer_pattern_num,train_sample_num(i)));
end
cancer_tissue_mask_down = 1-cancer_tissue_mask_down;
% cancer_tissue_mask_down = blkdiag(ones(cancer_pattern_num,train_sample_num(2)),...
%     ones(cancer_pattern_num,train_sample_num(3)),ones(cancer_pattern_num,train_sample_num(4)),...
%     ones(cancer_pattern_num,train_sample_num(5)),ones(cancer_pattern_num,train_sample_num(6)));

mask = [healthy_mask,[cancer_tissue_mask_top;cancer_tissue_mask_down]];

dot_W = repmat(weight_val',1,size(train_X,2));
% W = diag(W_value);

%% iteration
p=param.p; %L2-p
mu=param.mu;

% all_X = [train_X,test_X];
X = train_X;
R1 = 1/2*mu*norm(mask.*V,'fro')^2;
Z=dot_W.*(X-U*V);
err(1)=1/2*sum(sum(Z.^2,1).^(p/2)) + R1;
% err(1)=norm(Z,'fro')^2 + R1;
count=1;

while count<param.max_iterations
    count

    L2_p= p./(2*sum(Z.^2,1).^(1-p/2));
%     D_S=diag(L2_p);
    
    %% undate U
%     WXD =W*X*D_S;
    WXD = weight_val'*L2_p.*X;
    VD = V.*repmat(L2_p,size(V,1),1);
%     VD = V*D_S;
    
    factorU_numerator= WXD*V';
    factorU_denominator=dot_W(:,1:size(U,2)).*U*(VD*V');
    U=U.*(factorU_numerator./factorU_denominator);
    U(U>1) = 1.0-1e-4;
    
    %% update V
    factorV_numerator=U'*WXD ;
    factorV_denominator=U'*(dot_W(:,1:size(U,2)).*U)*VD+mu*mask.*V;
    V=V.*(factorV_numerator./factorV_denominator);
    norm_coff=sum(V,1);
    V=V./repmat(norm_coff,size(V,1),1);%normalization
    
    count=count+1;
    R1 = 1/2*mu*norm(mask.*V,'fro')^2;
    Z=dot_W.*(X-U*V);
    err(count)=1/2*sum(sum(Z.^2,1).^(p/2)) + R1;
%     err(count)=norm(Z,'fro')^2 + R1;
    err(count)
    if isnan(err(count))
        test = 0;
    end
    
    if abs(err(count-1)-err(count))<1e-2
        break;
    end
end

end

function weight_all = norm_hist(weight_all)
edges = 0:0.01:1;
prob_w = histcounts(weight_all,edges)/size(weight_all,2);
prob_cum = cumsum(prob_w);
new_w_bin = prob_cum*10;
weight_update = zeros(size(weight_all));
for i = 1:100
    update_index = ((i-1)*0.01<=weight_all) & (weight_all<i*0.01) & (weight_update <1);
    weight_all(update_index) = new_w_bin(i);
    weight_update(update_index) = 1;
end
end