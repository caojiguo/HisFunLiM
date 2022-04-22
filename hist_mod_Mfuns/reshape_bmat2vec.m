function [outvec] = reshape_bmat2vec(bmat, M)
K = (M+1)*(M+2)/2;
outvec = zeros(K, 1);
id0 = 1;
id1 = 1;
for j=1:(M+1)
    outvec(id0:id1) = bmat(M+2-j, 1:j);
    id0 = id0+j;
    id1 = id0+j;
end
end
