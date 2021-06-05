function [W,H,err]=sRFD(X, param)

K=size(X,1);
N=size(X,2);

class_num=param.class_num;
healthy_pattern_num = param.healthy_pattern_num;
cancer_pattern_num = param.cancer_pattern_num;
R=param.healthy_pattern_num + (class_num-1)*cancer_pattern_num; %
train_sample_num = param.train_sample_num;

%% Initialization

W=rand([K,R]); %
H=rand([R,N]);
norm_coff=sum(H,1);
H=H./repmat(norm_coff,size(H,1),1);%normalization

%% structural mask
healthy_top_mask = zeros(healthy_pattern_num, N);
healthy_left_mask = ones(R-healthy_pattern_num, train_sample_num(1));
cancer_mask = [];
for i = 1:size(train_sample_num,2)-1
    cancer_mask =  blkdiag(cancer_mask, ones(cancer_pattern_num, train_sample_num(i+1)));
end
cancer_mask = 1-cancer_mask;
train_mask = [healthy_left_mask,cancer_mask];
mask = [healthy_top_mask;train_mask];

%% parameters
p=param.p; %L2-p
lambda=param.lambda;

%% objective function
Omega = lambda*norm(mask.*H,'fro')^2
% norm(X-W*H,'fro')^2
% calculate_2p_norm(X-W*H,p) 
err(1)=calculate_2p_norm(X-W*H,p)  + Omega;

iter=1;
while iter<param.maximum_iterations 
    iter
    Z=X-W*H;
    L2_p= p./(2*sum(Z.^2,1).^(1-p/2));
    D=diag(L2_p);
    
    % update W
    factorW_numerator=X*D*H';
    factorW_denominator=W*(H*D*H');
    W=W.*(factorW_numerator./factorW_denominator);
    
    % update H
    factorH_numerator=W'*X*D ;
    factorH_denominator=W'*W*H*D+lambda*mask.*H;
    H=H.*(factorH_numerator./factorH_denominator);
    norm_coff=sum(H,1);
    H=H./repmat(norm_coff,size(H,1),1);%normalization
    
    iter=iter+1;
    Omega = lambda*norm(mask.*H,'fro')^2;
    err(iter) = calculate_2p_norm(X-W*H,p)  + Omega;
    err(iter)
    
    if abs(err(iter-1)-err(iter))<param.convergence_threshold
        break;
    end
end

end

function x_norm = calculate_2p_norm(X,p)
 x_norm = sum(sum(X.^2,1).^(p/2));
end