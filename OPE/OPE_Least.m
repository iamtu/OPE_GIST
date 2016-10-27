function [w,fun,time,iter] = opeLeast(X,y,lambda,theta,varargin)


% Non-convex optimization problem:
%
% min_w L(w) + \sum_i r_i(w)
%
% ============================ loss function ==============================
%
% L(w) = 1/(2n) \|Xw - y\|^2 (n: number of rows of X)
%
% ============================ regularizer ================================
% 
%  regtype = 2: Log Sum Penalty (LSP)
%            r_i(w) = lambda*\sum_i log(1 + |w_i|/theta), (theta > 0, lambda >= 0)
%
% ============================ Input ======================================
%
% X: data matrix with each row as a sample
%
% y: regression vector
%
% lambda: regularization parameter
%
% theta: theresholding parameter
%
% ======================= varargin: optional settings  ====================
%
% 'regtype': nonconvex regularization type 
%          1: CapL1 (default) 
%          2: LSP  
%          3: SCAD
%          4: MCP 
%
% 'stopcriterion': stopping criterion 
%                1: relative difference of objective functions 
%                   is less than tol (default)
%                0: relative difference of iterative weights is less
%                   than tol
%
% 'startingpoint': starting point (default: zero vector)
%
% 'tolerance': stopping tolerance (default: 1e-5)
%
% 'maxiteration': number of maximum iteration (default: 1000)
%
% 'tinitialization': initialization of t (default: 1)
%
% 'tmin': tmin parameter (default: 1e-20)
%
% 'tmax': tmax parameter (default: 1e-20)
%
% 'eta': eta factor (default: 2)
%
% 'sigma': parameter in the line search (default: 1e-5)
%
% 'nonmonotone': nonmonotone steps in the line search (default: 5)
% 
% 'stopnum': number of satisfying stopping criterion (default: 3)
%
% 'maxinneriter': number of maximum inner iteration (line search) (default: 20)
%
% ============================= Output ====================================
%
% w: output weight vector
%
% fun: a vector including all function values at each iteration
%
% time: a vector including all CPU times at each iteration
%
% iter: the number of iterative steps 
%
% =========================================================================

% default parameter settings
regtype = 1;
[n,d] = size(X);
w0 = zeros(d,1);
stopcriterion = 1;
tol = 1e-5; 
maxiter = 1000;

eta = 2;
stopnum = 3;
maxinneriter = 20;

% Initial function value

fun = zeros(maxiter+1,1); time = fun;
fun(1) = 0.5*w'*(grad - Xty/n) + 0.5*yty/n + funRegC(w,d,lambda,theta,regtype);
time(1) = 0;


[n,d] = size(X);
w0 = zeros(d,1);
s_t = zeros(d,1);
_a = 1;
w = w0; 
fun = zeros(maxiter+1,1); 
time = fun;


L = [0,0]
count = 0;
for iter = 1:maxiter
    tic;
    
    w_old = w;
    
    % chon ngau nhien g1, g2
    randIndex = randi([1,2],1);
    L(randIndex) += 1;

    % Tinh F'(w)

    dF =L(1) * (X'*(X*w - y))/n;
    for i = 1 : d 
        if w(i) > 0
            dF(i) += L(2) * lambda / (w(i) + theta);
        else if w(i) < 0
            dF(i) += L(2) * lambda / (w(i) - theta);
        else
    end


    % Tinh s_t = argmax<F'(w),x> x_i thuoc [-_a, _a] >
    for i = 1 : d 
        if dF(i) > 0
            s_t(i) = _a;
        else if dF(i) < 0
            s_t(i) = - _a;
        else
    end    

    w = w_old + (s_t - w_old) / iter;
    fun(iter+1) = 0.5*norm(X*w - y)^2/n + funRegC(w,d,lambda,theta,regtype);

    toc

    time(iter+1) = time(iter) + toc;


    % stopping condition
    if stopcriterion
        relativediff = abs(fun(iter) - fun(iter+1))/fun(iter+1);
    else
        relativediff = norm(w - w_old)/norm(w);
    end
    if relativediff < tol
        count = count + 1;
    else
        count = 0;
    end
    if count >= stopnum
        break;
    end
       
end 

fun = fun(1: min(maxiter,iter)+1);
time = time(1: min(maxiter,iter)+1);
