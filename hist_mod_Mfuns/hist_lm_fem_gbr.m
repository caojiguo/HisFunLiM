function [delta_gbr, bhat_gbr, BICs_gbr, RSQ]=hist_lm_fem_gbr(Xmat, Ymat, timevec, tau_vec, tuning_grids, M, bootstrap)
% This is a wrapper to implement nested group bridge approach
%   Xmat: nT-by-N
%   Ymat: nT-by-N
%   timevec: time points where X and Y are observed, nT-by-1 column,
%            rescale to [0,1]
%   

% Xmat = csvread('hip.csv') ;    % nT-by-N
% Ymat = csvread('knee.csv');    % nT-by-N
% timevec = linspace(0.025, T, nT)';   % time points of real observations

[nT, N] = size(Xmat);
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

xmeanfd = mean(xfd);
ymeanfd = mean(yfd);
yfd0 = center(yfd);
xfd0 = center(xfd);


% FEM basis
% the coordinates Si, Ti are counted between 0 and T
% M = nT-1;
lambda = (T-T0)/M;
K = (M+1)*(M+2)/2;

% Estimating the regresion function
npts = 2;
delta = lambda/(2*npts);
tpts = linspace(delta, T-delta, M*npts)'; % time points where y(t) and psi_{ik}(t) are evaluated
% size of psiMat = (N*M*npts) - by - K

% Y vector
yMat = eval_fd(yfd0,tpts)';   % size N-by-(M*npts)
yVect = reshape(yMat, N*M*npts, 1);
nsample = N*M*npts;

% create the design matrix using M
B = M;
eleNodes = NodeIndexation(M, B);
[Si, Ti] = ParalleloGrid(M, T, B);
psiMat = DesignMatrixFD(xfd0, npts, M, eleNodes, Si, Ti, B);
b0 = psiMat\yVect;

wH = tuning_grids.H;
wV = tuning_grids.V;
wP = tuning_grids.P;

% proposed approach
ntau = length(tau_vec);
BICs_gbr = nan(ntau, length(wH)); % set wH=wV=wP, in order to search less
bhat_gbr_temp = nan(K, ntau, length(wH));
tic
for itertau = 1:length(tau_vec)
    itertau
	for hh = 1:length(wH)
		w = wH(hh);
		[temp, ~, df_h] = bhat_gbridge(b0, yVect, psiMat, nsample, tau_vec(itertau), [w, w, w], 0.2, 100, M, K); % gamma=0.5, niter=100
		bhat_gbr_temp(:, itertau, hh) = temp;
		BICs_gbr(itertau, hh) = Mod_sel(yVect, temp, psiMat, nsample, df_h, 'BIC');
	end
end
toc
% ~400 secs
    est = (psiMat'*psiMat + pen_mat) \ psiMat' * yVect;
    Yhat = psiMat * est;
    RSS  = (yVect - Yhat)' * (yVect - Yhat);
    hat_mat = psiMat * inv(psiMat'*psiMat + pen_mat) * psiMat'; %#ok<MINV> ???
    df = trace(hat_mat);
    BIC_for_w(hh, vv, pp) = n*log(RSS/n) + log(n)*df ;

					
% manually avoids multiple minimum values of BIC_gbr
min_id = find(BICs_gbr==min(BICs_gbr), 1);
bhat_gbr = bhat_gbr_temp(:, min_id);
delta_gbr = calc_delta(reshape_bvec2mat(bhat_gbr, M), T);

if bootstrap == "true"
end


% computing fit to data
% compute the model approximation to the lip acceleration values 

% to calculate the intercept function: using the mean X curve as a model,
% and subtracting the fit that this gives from the mean Y
B = min_id - 1;
nfiney = 101;
tmesh = linspace(0,T,nfiney)';
psimeanArray = XMatrix(xmeanfd, tmesh, M, eleNodes, Si, Ti, B);
yHatMean = squeeze(psimeanArray)'*bhat_lma;
ymeanvec = eval_fd(ymeanfd, tmesh);
alphaHat = ymeanvec - yHatMean;


% final fit to data
% super-matrix containing the approximated lip acceleration curves
psiArray = XMatrix(xfd, tmesh, M, eleNodes, Si, Ti, B);
yHat1 = zeros(nfiney,N);

for i=1:N
    Xmati = squeeze(psiArray(i,:,:))';
    yHat1(:,i) = Xmati*bHat;
end

yHat = alphaHat*ones(1,N) + yHat1;

% residuals
resmat = ymat_plot - yHat;


% assessing the fit
% error sum of squares
SSE = sum(resmat.^2,2);

ymat0 = eval_fd(yfd0, tmesh);
SSY = sum(ymat0.^2,2);

RSQ = (SSY- SSE)./SSY;

end


