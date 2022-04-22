function [bhatgbr, df_s, df_h, RSS, df_s_new] = bhat_gbridge(b0, Y, U, n, tau, smooth_w, gamma, niter, M, K)
% smooth_w: (wH, wV, wP)

% Create the Amat containing grouping information
Amat = NaN(M+1,M+1);  
Amat(1, 1) = 1;
a0 = 2:3;
for j=2:(M+1)
    tmp = Amat(1:j, j:-1:1);
    tmp(logical(eye(j))) = a0;
    Amat(1:j, 1:j) = tmp(1:j, j:-1:1);
    a0 = a0(end) + (1:(j+1));
end

wH = smooth_w(1);
wV = smooth_w(2);
wP = smooth_w(3);
[DH, DV, DP] = create_penMat(M, M);


biter = zeros(length(b0), niter);
biter(:, 1) = b0;
for iterbg=2:niter
    % Step 1: compute the theta_js
    theta = zeros(M+1, 1);
    h     = zeros(M+1, 1);
    for j=1:(M+1)
        Aj_b_id = Amat( : , j:(M+1));
        Aj_b_id = Aj_b_id(~isnan(Aj_b_id));  
        
        % calculate cj 
        % b0 so that cj does not change over iteration?
        tempcj  = (M + 2 - j)^(1-gamma);
        bsm_Aj_norm_gamma  = sqrt(sum((b0(Aj_b_id)).^2))^gamma ;  
        cj = tempcj/bsm_Aj_norm_gamma;
        
        b_last = biter(Aj_b_id, iterbg-1);                     
        b2bargamma = (sum(abs(b_last)))^gamma;
        theta(j) = cj*((1/gamma-1)/tau)^gamma * b2bargamma ;
        
        h(j) = theta(j)^(1-1/gamma)*cj^(1/gamma) ;
    end
    
    % Step 2: compute the g_ks
    g = zeros(1, K);
    for k = 1:K
        [~, k_Acol_id] = find(Amat==k); % returns col number
        g(k) = sum(h(1: k_Acol_id)) ;
    end
    
    % Step 3: compute bs using glmnet
	% Defind a new design matrix, containing penalty matrices
    Utilde_gbr = [U; sqrt(wH)*DH; sqrt(wV)*DV; sqrt(wP)*DP] * diag(1./(n*g)) ;
	% And corresponding new Y vector
	Y_gbr = [Y; zeros(size(Utilde_gbr, 1)-length(Y), 1)];
    lassomodel = lasso(Utilde_gbr, Y_gbr, 'Lambda', 1/(2*n), 'Standardize', false); % returns a column vector
    % glmnet in R
    % glmnet(x = Ustarstargbr, y = Ystarbhatgbr, standardize = FALSE, alpha = 1, lambda = 0.5/n, family = "gaussian", intercept = FALSE)
    biter(: ,iterbg) =  lassomodel ./ (n*g)';    
   
    % Decide whether to stop the iteration
    difratio = zeros(1, length(biter(:,iterbg)));
    if iterbg >= 3 
        if isempty(find((biter(:,iterbg) - biter(:, (iterbg-1))) == 0, 1))  % if all bk_hat have changed
            difratio =  (biter(:,iterbg) - biter(:, (iterbg-1))) ./ biter(:,(iterbg-1));
        else
            idx1 = (biter(:,iterbg)-biter(:,(iterbg-1))) ~= 0;  % indicator of whether bk_hat has changed: 1=change, 0=no change
            difratio(idx1) = (biter(idx1, iterbg) - biter(idx1, (iterbg-1))) ./ biter(idx1,(iterbg-1)) ;
        end
        if max(difratio) < 10^(-4)
            break 
        end
    end
    
end

bhatgbr = biter(:, iterbg);

nsample = length(Y);
if isempty(find(bhatgbr == 0, 1))
    ula = U;
    w_diag = nsample * g./abs(bhatgbr) ;
else
    ula  = U(: , bhatgbr ~= 0);
    w_diag = nsample * g(bhatgbr ~= 0) ./ abs(bhatgbr(bhatgbr ~= 0)) ;
    DH2 = DH(:, bhatgbr ~= 0)' * DH(:, bhatgbr ~= 0);
    DV2 = DV(:, bhatgbr ~= 0)' * DV(:, bhatgbr ~= 0);
    DP2 = DP(:, bhatgbr ~= 0)' * DP(:, bhatgbr ~= 0);
end

% various DF's
% df_h, DF in Huang et al. (2009)
W_mat = diag(w_diag) ;
hat_h = ula * inv(ula'*ula + 0.5*W_mat) * ula'; %#ok<MINV>
df_h = trace(hat_h);

% df_s, DF in HLR paper, simpler than Huang (no W matrix)
hat_s = ula * inv(ula'*ula) * ula'; %#ok<MINV>
df_s = trace(hat_s);

% df_s_new, DF in HLR, corrected for adding the smoothness penalty
hat_s_new = ula * inv(ula'*ula + nsample*wH*DH2 + nsample*wV*DV2 + nsample*wP*DP2) * ula'; %#ok<MINV>
df_s_new = trace(hat_s_new);

Yhat = U*bhatgbr;
RSS = (Y - Yhat)' * (Y - Yhat) ;

end
