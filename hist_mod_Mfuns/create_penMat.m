function [DH_B, DV_B, DP_B]=create_penMat(M, B)
% generate the discrete penalty matrix
%   - first part: complete matrix when B=M
%   - second part: subsetting the corresponding sub-matrix for a given B

K = (M+1)*(M+2)/2;

% Create the Amat containing grouping information
% This is not needed to implement M&R(2003)
% But needed when imposing discrete penalty
Amat = NaN(M+1,M+1);  
Amat(1, 1) = 1;
a0 = 2:3;
for j=2:(M+1)
    tmp = Amat(1:j, j:-1:1);
    tmp(logical(eye(j))) = a0;
    Amat(1:j, 1:j) = tmp(1:j, j:-1:1);
    a0 = a0(end) + (1:(j+1));
end


% "full" DH, when B=M
% for B<M, extract corresponding rows and columns
DH = zeros(M*(1+M)/2, K);
ii = 1; % row indicator
tmp= 0;
for jj=2:K % column indicator
    if ismember(jj, Amat(1, 2:end))
        ii = ii+tmp;
        tmp = tmp + 1;
        for tmp_ii=1:tmp
            DH(ii+tmp_ii-1, jj+tmp_ii-1) = -1;
            DH(ii+tmp_ii-1, jj+tmp_ii) = 1;
        end
    end
end


% "full" DV, when B=M
% for B<M, extract corresponding rows and columns
DV = zeros(M*(1+M)/2, K);
ii = 1; % row indicator
tmp= 0;
for jj=2:K % column indicator
    if ismember(jj, Amat(1, 2:end))
        ii = ii+tmp;
        tmp = tmp + 1;
        for tmp_ii=1:tmp
            DV(ii+tmp_ii-1, ii+tmp_ii-1) = -1;
            DV(ii+tmp_ii-1, ii+tmp_ii+tmp-1) = 1;
        end
    end
end



% "full" DP, when B=M
% for B<M, extract corresponding rows and columns
DP = zeros(M*(1+M)/2, K);
ii = 1; % row indicator
tmp= 0;
for jj=2:K % column indicator
    if ismember(jj, Amat(1, 2:end))
        ii = ii+tmp;
        tmp = tmp + 1;
        for tmp_ii=1:tmp
            DP(ii+tmp_ii-1, ii+tmp_ii-1) = -1;
            DP(ii+tmp_ii-1, ii+tmp_ii+tmp) = 1;
        end
    end
end


if B==0 
    % discrete roughness penalty: parallel
    DP_B = DP(:, Amat(:, 1));
    DP_B = DP_B(all(DP_B==0, 2)~=1, :);
    DH_B = 0; % not used when B=0
    DV_B = 0; % not used when B=0
else
    % the id of the columns to be removed,
    % then the rows are removed accordingly
    no_col_id = reshape(Amat(:, (B+2):(M+1)), [], 1);
    no_col_id = no_col_id(~isnan(no_col_id));
    
    % horizontal penalty matrix
    DH_B = DH(:, ~ismember(1:K, no_col_id));
    DH_B = DH_B((sum(DH_B, 2)==0) & (all(DH_B==0, 2)~=1), :);
    
    % vertical penalty matrix
    DV_B = DV(:, ~ismember(1:K, no_col_id));
    DV_B = DV_B((sum(DV_B, 2)==0) & (all(DV_B==0, 2)~=1), :);
    
    % parallel penalty matrix
    DP_B = DP(:, ~ismember(1:K, no_col_id));
    DP_B = DP_B((sum(DP_B, 2)==0) & (all(DP_B==0, 2)~=1), :);
end



