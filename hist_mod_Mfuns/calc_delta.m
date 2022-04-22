function delta = calc_delta(bmat, T)
% delta: as the number of intervals going back in history
%        ranging from 0 to M
% search along lines / from (1,1), stop if / consists of non-zero values
M = size(bmat, 1)-1;
if bmat(1,1)~=0
    nint = M;
else
    for nint = M:(-1):0
        if nint==0
            break
        end
        ncol = M+2-nint;
        tmp = diag(bmat(1:ncol, ncol:(-1):1));        
        if sum(abs(tmp))~=0
            break
        end
    end
end
delta = nint*T/M;
end