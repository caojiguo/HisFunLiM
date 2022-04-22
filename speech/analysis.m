%% EMG and Lip

clear

% ... = proper path
addpath('.../Hist_FLM/hist_mod_Mfuns/')
addpath('.../Hist_FLM/hist_mod_Mfuns/fdaM')
addpath('.../Hist_FLM/Malfait_Ramsay_2003')

% load data
EMGmat = load ('EMG.dat');     % nT-by-N, x data
LIPmat = load ('LipAcc.dat');  % nT-by-N, y data
[nT, N] = size(EMGmat);
T = 0.69;
timevec = linspace(0,T,nT)';   % time points of real observations, used by M&R for interpolation only

tfine = (0:0.001:T)';          % for interpolation, smoothing to fd object, and plot
nfine = length(tfine);
lipmatlag = zeros(nfine-50,N);
EMGmatlag = zeros(nfine-50,N);
for i=1:N
    liptemp = interp1(timevec,LIPmat(:,i),tfine);
    EMGtemp = interp1(timevec,EMGmat(:,i),tfine);
    lipmatlag(:,i) = liptemp(51:nfine);
    EMGmatlag(:,i) = EMGtemp(1:(nfine-50));
end
nfine = nfine - 50;
tfine = tfine(1:nfine);
T = 0.64;


% B-spline basis for input data, smoothing the interpolated values into FD objects
nbasis = 93;
norder = 6;
basis = create_bspline_basis([0,T], nbasis, norder);
lipfd = data2fd(lipmatlag, tfine, basis);
emgfd = data2fd(EMGmatlag, tfine, basis);

emgmeanfd = mean(emgfd);
lipmeanfd = mean(lipfd);
lipfd0 = center(lipfd);
emgfd0 = center(emgfd);

% bivariate correlation function
nfiney = 101;
tmesh = linspace(0,T,nfiney)';
lipmat = eval_fd(lipfd, tmesh);
emgmat = eval_fd(emgfd, tmesh);
lipemgcorr = corrcoef([lipmat',emgmat']);
lipemgcorr = lipemgcorr(102:202,1:101);
for i=2:101, lipemgcorr(i,1:i-1) = nan; end

subplot(1,1,1)
colormap(hot)
surf(tmesh, tmesh, lipemgcorr', 'EdgeColor', 'none')
xlabel('\fontsize{16} s')
ylabel('\fontsize{16} t')
axis([0,.64,0,.64,-1,1])
axis('square')
colorbar
view([0,90])

% (full) FEM basis
M = 20;
lambda = T/M;
B = M;
K = (M+1)*(M/2+1);
eleNodes = NodeIndexation(M, B);
[Si, Ti] = ParalleloGrid(M, T, B);


% Calculate the design matrix (Psi in paper)
% npts: ~ number of points in each sub-interval for integral approximation 
npts = 8;
delta = lambda/(2*npts);
tpts = linspace(delta, T-delta, M*npts)'; % time points where y(t) and psi_{ik}(t) are evaluated

% size of psiMat = (N*M*npts) - by - K
% K = (M+1)*(M+2)/2
psiMat_full = DesignMatrixFD(emgfd0, npts, M, eleNodes, Si, Ti, B); % ERROR !!!
% the check range statment is commented out in order to run the program

% check for singularity
singvals = svd(full(psiMat_full));
condition = max(singvals)/min(singvals);
disp(['Condition number = ',num2str(condition)])

% Y vector
yMat = eval_fd(lipfd0,tpts)';   % size N-by-(M*npts)
yVect = reshape(yMat, N*M*npts, 1);

n = length(yVect);
SSY = yVect' * yVect;



%% M=20, 
% GRB approach (with discrete PEN)
tau = exp(linspace(-17,-15,5));   % M=30, ~15 mins 
wH_gbr = exp(linspace(-10,-1, 10)); 
gamma = 0.5;
niter = 500;

tic
bhat_gbr_temp = zeros(K, length(tau), length(wH_gbr));
BICs_gbr = zeros(length(tau), length(wH_gbr));
SSE_gbr = zeros(length(tau), length(wH_gbr));
b0 = psiMat_full\yVect;
for itertau = 1:length(tau)
    itertau
    for hh = 1:length(wH_gbr)
        [temp, ~, df_h, RSS] = bhat_gbridge(b0, yVect, psiMat_full, n, tau(itertau), [wH_gbr(hh),wH_gbr(hh),wH_gbr(hh)], gamma, niter, M, K);
        bhat_gbr_temp(:, itertau, hh) = temp;
        BICs_gbr(itertau, hh) = Mod_sel(RSS, n, df_h, 'BIC');        
        SSE_gbr(itertau, hh)  = RSS;
    end
end

% manually avoids multiple minimum values of BIC_gbr
[~, min_id] = min(BICs_gbr(:));
[I_tau, I_w] = ind2sub(size(BICs_gbr), min_id);
bhat_gbr = bhat_gbr_temp(:, I_tau, I_w);
delta_gbr = calc_delta(reshape_bvec2mat(bhat_gbr, M), T);

% Refit step: after BIC/AIC selection, determine delta and Bsim, re-est b
B_est = round(delta_gbr/lambda);
eleNodes = NodeIndexation(M, B_est);
[Si, Ti] = ParalleloGrid(M, T, B_est);
psiMat = DesignMatrixFD(emgfd0, npts, M, eleNodes, Si, Ti, B_est);
bhat_gbr = psiMat\yVect;

Yhat = psiMat * bhat_gbr;
SSE_final = (yVect - Yhat)' * (yVect - Yhat);

format short g; [B_est, delta_gbr, I_tau, SSE_final, 1-SSE_final/SSY,  min(BICs_gbr(:))]

toc


%% bootstrap for beta for fixed delta
nt_plot = length(svec);
betaMat_boots = nan(nt_plot, nt_plot, nboots);
eleNodes = NodeIndexation(M, B_est);
[Si, Ti] = ParalleloGrid(M, T, B_est);
K_Bp = (B_est+1)*(M+1-B_est/2);
psiMat_forbetaBoot = DesignMatrixFD(emgfd0, npts, M, eleNodes, Si, Ti, B_est);
[DH_B, DV_B, DP_B] = create_penMat(M, B_est);
DH_B2 = DH_B' * DH_B;
DV_B2 = DV_B' * DV_B;
DP_B2 = DP_B' * DP_B;
pen_mat = wH_gbr*DH_B2 + wH_gbr*DV_B2 + wH_gbr*DP_B2;
rng(10000);
for ii=1:nboots
    if mod(ii, 10)==1
        ii
    end
    yboots = Yhat + normrnd(0, sigma, [n, 1]);
    est = (psiMat_forbetaBoot'*psiMat_forbetaBoot + pen_mat) \ psiMat_forbetaBoot' * yboots;
    betaMat = BetaEvalFD(svec, tvec, est, M, T, lambda, eleNodes, Si, Ti, B_est);
    betaMat(betaMat==0)=NaN;
    betaMat_boots(:, :, ii) = betaMat;
end
betahat_se = std(betaMat_boots, 0, 3, 'omitnan');  % same size as betaMat, nt_plot^2
subplot(1,1,1)
surf(svec, tvec, betaMat0./betahat_se, 'EdgeColor', 'none')
xlabel('\fontsize{16} s (msec)')
ylabel('\fontsize{16} t (msec)')
axis([0,T,0,T])
% axis('square')
view([0, 90]);colormap(hot);colorbar






