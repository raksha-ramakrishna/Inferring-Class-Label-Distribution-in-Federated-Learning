function X = SimplexProj(Y)
[N,D] = size(Y);
Y(isinf(Y))=1;
X = sort(Y,2,'descend');
Xtmp = (cumsum(X,2)-1)*diag((1./sparse(1:D)));
X = max(bsxfun(@minus,Y,Xtmp(sub2ind([N,D],(1:N)',sum(X>Xtmp,2)))),0);
end