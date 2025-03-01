function [P3,t,ob]=fgsvd(A,B,b)
tol=1e-4;
[Q1,t,ob]=myqr([A',B'],tol,b);
G=A*Q1;
H=B*Q1;
[L,R]=qr([G;H],0);
L1=L(1:size(A,1),:);
[W1,~]=eig(L1'*L1);
P3=W1'*R*Q1';
end
