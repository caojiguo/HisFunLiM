function bhat_lma_long = reshape_bhat_lma_short2long(bhat_lma_short, M)
% input a bhat by lma method, of length Kshort (<=K, depending on B)
% output a long bhat of length K, with 0's properly inserted
K = (M+1)*(M+2)/2;
Kshort = length(bhat_lma_short);
B = M+1/2-sqrt((2*M+1)^2-8*(Kshort-M-1))/2;

% convert bhat_lma_short to matrix form
id0 = 1;
bhat_lma_mat = nan(M+1, M+1);
for j=1:(M+1)
    if j<=(B+1)
        bhat_lma_mat(M+2-j, 1:j) = bhat_lma_short(id0:(id0+j-1));
        id0 = id0+j;
    else
        bhat_lma_mat(M+2-j, 1:(j-B-1)) = 0;
        bhat_lma_mat(M+2-j, (j-B):j) = bhat_lma_short(id0:(id0+B));
        id0 = id0+B+1;
    end    
end

bhat_lma_long = reshape_bmat2vec(bhat_lma_mat, M);
end
