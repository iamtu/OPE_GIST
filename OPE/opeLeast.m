function [w,fun,time,iter,fun_min] = opeLeast(X,y,lambda,theta,varargin)


% OPE to solve Non-convex optimization problem:
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
% 'startingpoint': starting point (default: zero vector)
%
% 'maxiteration': number of maximum iteration (default: 1000)
% 
% 'bound' : bound value of w
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
% fun_min : the minimum value OPE can reach
%
% =========================================================================

[n,d] = size(X);
w0 = zeros(d,1);
maxiter = 100;
a = 1;
epsilon = 1e-6;

% Optional parameter settings
parameterCount = length(varargin)/2;

for parameterIndex = 1:parameterCount,
    parameterName = varargin{parameterIndex*2 - 1};
    parameterValue = varargin{parameterIndex*2};
    switch lower(parameterName)
        case 'regtype'
            regtype = parameterValue;
            if regtype == 3 && theta <= 2 
                error('\theta must be greater than 2!');
            end
        case 'startingpoint'
            w0 = parameterValue;
        case 'maxiteration'
            maxiter = parameterValue;
        case 'bound'
        	a = parameterValue;
        case 'epsilon'
            epsilon = parameterValue;
        otherwise
            error(['The parameter ''' parameterName ''' is not recognized by the function ''' mfilename '''!']);
    end
end

fprintf('OPE params : regtype = %d, maxiter = %d, bound = %f, epsilon = %f\n', regtype, maxiter, a, epsilon);

s_t = zeros(d,1);
w = w0; 
fun = zeros(maxiter+1,1); 
time = fun;
time(1) = 0;
fun(1) = 0.5*norm(X*w - y)^2/n + funRegC(w,d,lambda,theta,regtype);
fun_min = fun(1);
L = [1,1];

for iter = 1:maxiter

    tic;
    
    w_old = w;
    
    % chon ngau nhien g1, g2
    randIndex = randi([1,2],1);
    L(randIndex) = L(randIndex) + 1;
    
    % Tinh F'(w)

    dF = L(1) * (X'*(X*w_old - y))/n + L(2) * derRegC(w_old,d,lambda,theta, epsilon,regtype);
    
    s_t = findDirection(dF, d, a);

    alpha_ = 2/(iter + 2);
    w = w_old + (s_t - w_old) * alpha_;
    
    
    fun(iter+1) = 0.5*norm(X*w - y)^2/n + funRegC(w,d,lambda,theta,regtype);

    time(iter+1) = time(iter) + toc;
    if (fun(iter+1) < fun_min)
        fun_min = fun(iter+1);
    end
    
end 
