function [delta_lma, bhat_lma, BICs_lma, RSQ]=hist_lm_fem_lma(Xmat, Ymat, timevec, tuning_grids, M, corrplot)
% This is a wrapper to implement M&R(2003) 
%   Xmat: nT-by-N
%   Ymat: nT-by-N
%   timevec: time points where X and Y are observed, nT-by-1 column,
%            rescale to [0,1]
%   tuning_grids: a structure, containing values of wH, wV, wP for grid search

% Xmat = csvread('hip.csv') ;    % nT-by-N
% Ymat = csvread('knee.csv');    % nT-by-N
% timevec = linspace(0.025, T, nT)';   % time points of real observations

[~, N] = size(Xmat);
T = timevec(end);  
T0 = 0;

% for interpolation, smoothing to fd object
tfine = (T0:0.001:T)';          
nfine = length(tfine);
Ymatfine = zeros(nfine, N);
Xmatfine = zeros(nfine, N);
for i=1:N
    Ymatfine(:,i) = interp1(timevec, Ymat(:,i), tfine);
    Xmatfine(:,i) = interp1(timevec, Xmat(:,i), tfine);
end

% B-spline basis for input data, smoothing the interpolated values into FD objects
nbasis = 90;
norder = 6;
basis = create_bspline_basis([0,T], nbasis, norder);
yfd = data2fd(Ymatfine, tfine, basis);
xfd = data2fd(Xmatfine, tfine, basis);

% xmeanfd = mean(xfd);
% ymeanfd = mean(yfd);
yfd0 = center(yfd);
xfd0 = center(xfd);

% bivariate correlation function
nfiney = 101;
tmesh = linspace(T0,T,nfiney)';
ymat_plot = eval_fd(yfd, tmesh);
xmat_plot = eval_fd(xfd, tmesh);
if corrplot=="true"
    xycorr = corrcoef([ymat_plot',xmat_plot']);
    xycorr = xycorr(102:202,1:101);
    for i=2:101, xycorr(i,1:i-1) = 0; end
    
    subplot(1,1,1)
    colormap(hot)
    surf(tmesh, tmesh, xycorr')
    xlabel('\fontsize{16} s')
    ylabel('\fontsize{16} t')
    axis([T0, T, T0, T, -1, 1])
    axis('square')
end

% FEM basis
% the coordinates Si, Ti are counted between 0 and T
lambda = (T-T0)/M;
K = (M+1)*(M+2)/2;

% Estimating the regresion function
npts = 20;
delta = lambda/(2*npts);
tpts = linspace(delta, T-delta, M*npts)'; % time points where y(t) and psi_{ik}(t) are evaluated
% size of psiMat = (N*M*npts) - by - K

% Y vector
yMat = eval_fd(yfd0,tpts)';   % size N-by-(M*npts)
yVect = reshape(yMat, N*M*npts, 1);
 

wH = tuning_grids.H;
wV = tuning_grids.V;
wP = tuning_grids.P;

% M&R (2003) approach + discrete penalty
n = length(yVect);
BICs_lma = zeros(M+1, 1);
bhat_lma_temp = zeros(K, M+1);
SSE_temp = zeros(M+1, 1);
tic
for B=0:M
    B
    % create the design matrix using M
    eleNodes = NodeIndexation(M, B);
    [Si, Ti] = ParalleloGrid(M, T, B);
    K_B = (B+1)*(M+1-B/2);
    psiMat = DesignMatrixFD(xfd0, npts, M, eleNodes, Si, Ti, B);
    % b0 is the estimated coefficient vector 
    if B==0
        [~, ~, DP_B]=create_penMat(M, B);
        DP_B2 = DP_B' * DP_B;
        BIC_for_w = nan(1, length(wP));
        for pp=1:length(wP) 
            pen_mat = wP(pp)*DP_B2;
            est = (psiMat'*psiMat + pen_mat) \ psiMat' * yVect;
            Yhat = psiMat * est;
            RSS  = (yVect - Yhat)' * (yVect - Yhat);
            hat_mat = psiMat * inv(psiMat'*psiMat + pen_mat) * psiMat'; %#ok<MINV> ???
            df = trace(hat_mat);
            BIC_for_w(pp) = n*log(RSS/n) + log(n)*df ;
        end
        [~, min_id] = min(BIC_for_w);
        b0 = (psiMat'*psiMat + wP(min_id)*DP_B2) \ psiMat' * yVect;    
        BICs_lma(B+1, 1) = BIC_for_w(min_id); 
    else
        [DH_B, DV_B, DP_B] = create_penMat(M, B);
        DH_B2 = DH_B' * DH_B;
        DV_B2 = DV_B' * DV_B;
        DP_B2 = DP_B' * DP_B;
        % straightforward calculation of bhat, via a grid search
        BIC_for_w = nan(length(wH), length(wV), length(wP));
        for hh=1:length(wH)
            for vv=1:length(wV)
                for pp=1:length(wP)
                    pen_mat = wH(hh)*DH_B2 + wV(vv)*DV_B2 + wP(pp)*DP_B2;
                    est = (psiMat'*psiMat + pen_mat) \ psiMat' * yVect;
                    Yhat = psiMat * est;
                    RSS  = (yVect - Yhat)' * (yVect - Yhat);
                    hat_mat = psiMat * inv(psiMat'*psiMat + pen_mat) * psiMat'; %#ok<MINV> ???
                    df = trace(hat_mat);
                    BIC_for_w(hh, vv, pp) = n*log(RSS/n) + log(n)*df ;
                end
            end
        end
        [~, min_id] = min(BIC_for_w(:));
        [I_h, I_v, I_p] = ind2sub(size(BIC_for_w), min_id);   
        pen_mat = wH(I_h)*DH_B2 + wV(I_v)*DV_B2 + wP(I_p)*DP_B2;
        b0 = (psiMat'*psiMat + pen_mat) \ psiMat' * yVect;
        BICs_lma(B+1, 1) = BIC_for_w(I_h, I_v, I_p); 
    end    
    
    Yhat = psiMat * b0;
    SSE_temp(B+1, 1) = (yVect - Yhat)' * (yVect - Yhat) ;
%     BICs_lma(B+1, 1) = nsample * log(SSE_temp(B+1, 1)/nsample) + log(nsample)* K_B;
    % reshape to vector of length K, with 0 properly inserted
    bhat_lma_temp(:, B+1) = reshape_bhat_lma_short2long(b0, M);
end
toc
% ~1600 sec for npts=4
% ~800 sec for npts=2 <for a problem tried>

min_id = find(BICs_lma == min(BICs_lma), 1); % manually avoids dimension mismatch
bhat_lma = bhat_lma_temp(:, min_id);
delta_lma = calc_delta(reshape_bvec2mat(bhat_lma, M), T);
SSE = SSE_temp(min_id);

% computing fit to data
% compute the model approximation to the lip acceleration values 

% to calculate the intercept function: using the mean X curve as a model,
% and subtracting the fit that this gives from the mean Y
% B = min_id - 1;
% nfiney = 101;
% tmesh = linspace(0,T,nfiney)';
% psimeanArray = XMatrix(xmeanfd, tmesh, M, eleNodes, Si, Ti, B);
% yHatMean = squeeze(psimeanArray)'*bhat_lma(bhat_lma~=0);
% ymeanvec = eval_fd(ymeanfd, tmesh);
% alphaHat = ymeanvec - yHatMean;


% final fit to data
% super-matrix containing the approximated y curves
% psiArray = XMatrix(xfd0, tmesh, M, eleNodes, Si, Ti, B);
% yHat0 = zeros(nfiney,N);

% for i=1:N
%     Xmati = squeeze(psiArray(i,:,:))';
%     yHat0(:,i) = Xmati*bhat_lma(bhat_lma~=0);
% end

% residuals
% ymat0 = eval_fd(yfd0, tmesh);
% SSY = sum(ymat0.^2,2);
% resmat = ymat0 - yHat0 ;

% assessing the fit
% error sum of squares
% SSE = sum(resmat.^2,2);
SSY = (yVect )' * (yVect  );

% per time point
RSQ = (SSY- SSE)./SSY;

end


