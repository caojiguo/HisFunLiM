%% scenario 1:
%  beta(s,t) is constant over the trapezoidal region

clear

% set-up proper path 
addpath('.../Hist_FLM/hist_mod_Mfuns')
addpath('.../Hist_FLM/hist_mod_Mfuns/fdaM')

simu_id = 1;
M = 20;      % M=2, is simu4 case
Msim = 20;   % prefer a large M, as it decides final choice of delta
nsim = 100;

%% generate data: truth
% triangulation over the full domain by setting B=M
T = 1;
B = M;
K = (M+1)*(M+2)/2; % 
lambda = T/M;
eleNodes = NodeIndexation(M, B);
[Si, Ti] = ParalleloGrid(M, T, B);

% Create coefficients according to triangular position
deltaM = round(M/2);
bmat = NaN(M+1,M+1);
p1 = 1;
for j=1:(M+1)
    tmp = bmat(1:j, j:-1:1);
    if j<=(M-deltaM+1)
        tmp(logical(eye(j))) = 0;
    else
        tmp(logical(eye(j))) = binornd(1, p1, j, 1);
    end
    bmat(1:j, 1:j) = tmp(1:j, j:-1:1);
end

% stretch from matrix to vector form
bvec_true = reshape_bmat2vec(bmat, M)*10;

delta_true = calc_delta(reshape_bvec2mat(bvec_true, M), T);

% plot of beta(s,t)
% mesh grid over which to evaluate beta(s,t), true and estimated
nspt = 101;
svec = linspace(0, T, nspt)';
tvec = linspace(0, T, nspt)';
betaMat_true = BetaEvalFD(svec, tvec, bvec_true, M, T, lambda, eleNodes, Si, Ti, B);
for ii = 1:nspt
    for jj=(ii+1):nspt
        betaMat_true(ii, jj) = NaN;
    end
end
surf(svec, tvec, betaMat_true, 'EdgeColor', 'none')

for ii = 1:nspt
    for jj=(ii+1):nspt
        betaMat_true(ii, jj) = 0;
    end
end

% load x data
Xmat = load ('simu_x.dat');
[nt, N] = size(Xmat);
timevec = linspace(0, T, nt)';
Xmat = Xmat - repmat(mean(Xmat, 2), 1, N);

tfine = (0:0.001:T)';
nfine = length(tfine);
Xmatfine = zeros(nfine, N);
for i=1:N
    Xmatfine(:,i) = interp1(timevec, Xmat(:,i),tfine);
end

% B-spline basis for input data
nbasis = 93;
norder = 6;
basis = create_bspline_basis([0,T], nbasis, norder);
xfd = data2fd(Xmatfine, tfine, basis);

% create the design matrix, and generate yi(tj)
psiArray = XMatrix(xfd, timevec, M, eleNodes, Si, Ti, B);
ymeanmat = zeros(nt, N);
for i =1:N
    ymeanmat(:, i) = squeeze(psiArray(i, :, :))'* bvec_true ;
end


%% Tuning parameters
% discrete penalty
wH_lma = exp(linspace(-8, -6, 5));

% seq of penalty parameters for GBR approach
tau = exp(linspace(-22, -21, 8));
wH_gbr = exp(linspace(-16, -14, 5));

%% Simulation
sd = 0.5;

Ksim = (Msim+1)*(Msim+2)/2;
lambdasim = T/Msim;

% results of interest
bhat_lma = nan(Ksim, nsim); % lma = penalized linear model approximation approach (PLMA)
bhat_gbr = nan(Ksim, nsim); % gbr = nested group bridge approach (NGB)
delta_est = nan(nsim, 2);   % columns: PLMA, NGB
% mean integrated squared error of beta_hat(s,t)
MISE = nan(nsim, 2);
SSE_lma_gbr = nan(nsim, 2); % final fit 
Rsq_lma_gbr = nan(nsim, 2); % final fit 

% more parameters for GBR approach
gamma = 0.5;
niter = 500;

% temp storage during each repetition
bhat_lma_temp = zeros(Ksim, Msim+1); % used to select a final estimate
bhat_gbr_temp = zeros(Ksim, length(tau), length(wH_gbr));
BICs_lma = zeros(Msim+1, 1);
BICs_gbr = zeros(length(tau), length(wH_gbr));

% psiMat's corresponding to Bsim=0, 1, ..., M, for repeated use later
% saved in an array of dim (M+1)x(M*npts*N)xK
% divide each subintervel into 4 piecies for integral approximation
npts = 4;
ntpts = Msim*npts;
psiMat_all = nan(Msim+1, ntpts*N, Ksim);

condition_num = nan(Msim+1, 1);
for Bsim=0:Msim
    % create the design matrix using Msim
    eleNodes = NodeIndexation(Msim, Bsim);
    [Si, Ti] = ParalleloGrid(Msim, T, Bsim);
    K_Bsim = (Bsim+1)*(Msim+1-Bsim/2);
    tic
    psiMat_all(Bsim+1, :, 1:K_Bsim) = DesignMatrixFD(xfd, npts, Msim, eleNodes, Si, Ti, Bsim);
    
    % check for singularity
    psiMat = squeeze(psiMat_all(Bsim+1, :, 1:K_Bsim)) ;
    singvals = svd(full(psiMat));
    condition_num(Bsim+1) = max(singvals)/min(singvals);
end

rng(123);
for ns = 1:nsim
    
    if mod(ns, 10)==1 || ns==nsim
        ns
    end
    
    % add i.i.d. epsilon(t)
    Ymat = ymeanmat + normrnd(0, sd, size(ymeanmat));
        
    % create fd object for yi(t)'s
    yfine = zeros(nfine, N);
    for i=1:N
        yfine(:,i) = interp1(timevec, Ymat(:,i), tfine);
    end
    yfd = data2fd(yfine, tfine, basis);
    
    % Y vector, created from above yfd
    dd = lambdasim/(2*npts);
    tpts = linspace(dd, T-dd, ntpts)';
    yMat = eval_fd(yfd,tpts)';
    Y = reshape(yMat, N*Msim*npts, 1);
    SSY = Y'*Y;
    nsample = length(Y);
    
    % M&R (2003) + discrete penalty = Harzelak et al. (2007)
    % loop over Bsim = 0, 1, ..., Msim-2, Msim-1, Msim ( delta/lamda = 1, 2, ..., M-1, M, M)
    % for each Bsim, calculate psiMat, and solve for bvec
    % then use BIC to select a final estimate
    for Bsim = 0:Msim
        if Bsim==round(Msim/2)
            Bsim;
        end
        K_Bsim = (Bsim+1)*(Msim+1-Bsim/2);
        psiMat = squeeze(psiMat_all(Bsim+1, :, 1:K_Bsim));
        if Bsim==0
            [~, ~, DP_B]=create_penMat(Msim, Bsim);
            DP_B2 = DP_B' * DP_B;
            BIC_for_w = nan(1, length(wH_lma));
            for pp=1:length(wH_lma)
                pen_mat = wH_lma(pp)*DP_B2;
                est = (psiMat'*psiMat + pen_mat) \ psiMat' * Y;
                Yhat = psiMat * est;
                RSS  = (Y - Yhat)' * (Y - Yhat);
                hat_mat = psiMat * inv(psiMat'*psiMat + pen_mat) * psiMat'; %#ok<MINV> 
                df = trace(hat_mat);
                BIC_for_w(pp) = nsample*log(RSS/nsample) + log(nsample)*df ;
            end
            [~, min_id] = min(BIC_for_w);
            b0 = (psiMat'*psiMat + wH_lma(min_id)*DP_B2) \ psiMat' * Y;
            BICs_lma(Bsim+1, 1) = BIC_for_w(min_id);
        else
            [DH_B, DV_B, DP_B] = create_penMat(Msim, Bsim);
            DH_B2 = DH_B' * DH_B;
            DV_B2 = DV_B' * DV_B;
            DP_B2 = DP_B' * DP_B;
            % straightforward calculation of bhat, via a grid search
            BIC_for_w = nan(1, length(wH_lma));
            for hh=1:length(wH_lma)
                pen_mat = wH_lma(hh)*DH_B2 + wH_lma(hh)*DV_B2 + wH_lma(hh)*DP_B2;
                est = (psiMat'*psiMat + pen_mat) \ psiMat' * Y;
                Yhat = psiMat * est;
                RSS  = (Y - Yhat)' * (Y - Yhat);
                hat_mat = psiMat * inv(psiMat'*psiMat + pen_mat) * psiMat'; %#ok<MINV> ???
                df = trace(hat_mat);
                BIC_for_w(hh) = nsample*log(RSS/nsample) + log(nsample)*df ;
            end
            [~, min_id] = min(BIC_for_w(:));
            pen_mat = wH_lma(min_id)*DH_B2 + wH_lma(min_id)*DV_B2 + wH_lma(min_id)*DP_B2;
            b0 = (psiMat'*psiMat + pen_mat) \ psiMat' * Y;
            BICs_lma(Bsim+1, 1) = BIC_for_w(min_id);
        end
       
        Yhat = psiMat * b0;
        SSE = (Y - Yhat)' * (Y - Yhat) ;
        BICs_lma(Bsim+1, 1) = nsample * log(SSE/nsample) + log(nsample)* K_Bsim;
        % reshape to vector of length K, with 0 properly inserted
        bhat_lma_temp(:, Bsim+1) = reshape_bhat_lma_short2long(b0, Msim);
    end
    min_id = find(BICs_lma(:, 1) == min(BICs_lma(:, 1)), 1) ; % manually avoids dimension mismatch
    bhat = bhat_lma_temp(:, min_id);
    bhat_lma(:, ns, 1) = bhat;
    delta_est(ns, 1) = calc_delta(reshape_bvec2mat(bhat, Msim), T);
    Yhat = psiMat * bhat;
    SSE = (Y - Yhat)' * (Y - Yhat) ;
    SSE_lma_gbr(ns, 1) = SSE;
    Rsq_lma_gbr(ns, 1) = 1 - SSE/SSY;
    
    
    % GBR: loop over tau's values
    % GBR takes information from Bsim=Msim above: psiMat, b0
    % psiMat is the same for all tau's
    for itertau = 1:length(tau)
        for hh = 1:length(wH_gbr)
            [temp, ~, ~, RSS, df_s_new] = bhat_gbridge(b0, Y, psiMat, nsample, tau(itertau), [wH_gbr(hh),wH_gbr(hh),wH_gbr(hh)], gamma, niter, Msim, Ksim);
            bhat_gbr_temp(:, itertau, hh) = temp;
            % BIC
            BICs_gbr(itertau, hh) = nsample*log(RSS/nsample) + log(nsample)*df_s_new;
        end
    end
    % manually avoids multiple minimum values of BIC_gbr
    [~, min_id] = min(BICs_gbr(:));
    [I_tau, I_w] = ind2sub(size(BICs_gbr), min_id);
    bhat = bhat_gbr_temp(:, I_tau, I_w);
    delta_est(ns, 2) = calc_delta(reshape_bvec2mat(bhat, Msim), T);
    % final fit assessment using BIC
    Yhat = psiMat * bhat;
    SSE = (Y - Yhat)' * (Y - Yhat) ;
    SSE_lma_gbr(ns, 2) = SSE;
    Rsq_lma_gbr(ns, 2) = 1 - SSE/SSY;
    % Refit step: after BIC/AIC selection, determine delta and Bsim, re-est b
    B_est = round(delta_est(ns, 2)/lambdasim);
    bhat_gbr(:, ns) = bhat_lma_temp(:, B_est+1);
    
    % error of beta_hat(s,t) and MISE
    betaMat_lma = BetaEvalFD(svec, tvec, bhat_lma(:, ns), Msim, T, lambda, eleNodes, Si, Ti, Bsim);
    MISE(ns, 1) = 2/nspt/(nspt+1)*sum(sum((betaMat_lma - betaMat_true).^2));
    
    betaMat_gbr = BetaEvalFD(svec, tvec, bhat_gbr(:, ns), Msim, T, lambda, eleNodes, Si, Ti, Bsim);
    MISE(ns, 2) = 2/nspt/(nspt+1)*sum(sum((betaMat_gbr - betaMat_true).^2));
    
end


%% Quick outputs
mean_MISE_beta_hat = mean(MISE)
sd_MISE_beta_hat = std(MISE)

mean_delta = mean(delta_est, 1)
bias_delta_perc = (mean(delta_est, 1) - delta_true)./ delta_true .* 100
sd_delta = std(delta_est, 1)
rootMSE_delta = sqrt((mean(delta_est, 1) - delta_true).^2 + var(delta_est, 1))

mean_SSE = mean(SSE_lma_gbr, 1)
mean_Rsq = mean(Rsq_lma_gbr, 1)


