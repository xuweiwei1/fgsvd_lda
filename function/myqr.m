function [Q1,t,ob] = myqr(A,tol,b)%,t,ob     (使用的)
n = size(A,2);
Y = A*randn(n,b);
[Q,~] = qr(Y,0);
Q1 = [];
ny = 1;
t = 0;
while ny>tol
    t = t + 1;
    Q1 = [Q1,Q];
    M = Q1'*Q1;
    index = find(abs(M(:,1)) > 1e-5, 2, 'first');
    if numel(index) < 2
        clear index
    else
        Q1=Q1(:,1:index(2)-1);
            break;
    end
    ob(t) = norm(A-Q1*(Q1'*A),'fro');
    Y = A*randn(n,b);
    Y = Y -Q1*(Q1'*Y);
    [Q,R] = qr(Y,0);
end
end


