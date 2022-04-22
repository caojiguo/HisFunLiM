function [outmat] = reshape_bvec2mat(bvec, M)
  outmat = nan(M+1, M+1);
  id0 = 1;
  id1 = 1;
  for j=1:(M+1)
    outmat(M+2-j, 1:j) = bvec(id0:id1);
    id0 = id0+j;
    id1 = id0+j;
  end
end
