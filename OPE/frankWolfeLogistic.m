function [w,fun,time,iter,fun_min] = frankWolfeLogistic(X,y,lambda,theta,varargin)
% OPE to solve Logsitic Regression loss
%
% Non-convex optimization problem:
%
% min_w L(w) + \sum_i r_i(w)
%
% ================================ loss function ==========================
%
% L(w) = 1/n \sum_j log(1 + exp(-y_j*x_j'*w)) (n: number of samples)
%
% ================================ regularizer ============================
%
%  regtype = 1: Capped L1 regularizer (CapL1) (default) 
%            r_i(w) = lambda*\min(|w_i|,theta), (theta > 0, lambda >= 0)
% 
%  regtype = 2: Log Sum Penalty (LSP)
%            r_i(w) = lambda*\sum_i log(1 + |w_i|/theta), (theta > 0, lambda >= 0)
% 
%  regtype = 3: Smoothly Clipped Absolute Deviation (SCAD)
%            r_i(w) = lambda*|w_i|, if |w_i|<=lambda
%            r_i(w) = (-w_i^2 + 2*theta*lambda*|w_i| - lambda^2)/(2(theta - 1)), if lambda<=|w_i|<=theta*lambda
%            r_i(w) = 0.5*(theta + 1)*lambda^2, if |w_i| > theta*lambda, (theta > 2, lambda >= 0)
%
%  regtype = 4: Minimax Concave Penalty (MCP)
%            r_i(w) = lambda*|w_i| - 0.5*w_i^2/theta, if |w_i|<=theta*lambda
%            r_i(w) = 0.5*theta*lambda^2, if |w_i| > theta*lambda, (theta >
%            0, lambda >= 0)
%
% ============================ Input ======================================
%
% X: data matrix with each row as a sample
%
% y: label vector (+1 or -1)
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
% 'bound' : bound value
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
% fun_min : the minimum value OPE can reach in iterations


if nargin < 4
    error('Too few input parameters!');
end

if theta <= 0 || lambda < 0
    error('\theta must be positive and \lambda must be nonneagtive!');
end

% Parse the optional inputs.
if (mod(length(varargin), 2) ~= 0 ),
    error(['Optional Parameters passed to the function ''' mfilename ''' must be passed in pairs!']);
end

% default parameter settings
regtype = 1;
[n,d] = size(X); 
w0 = zeros(d,1);
maxiter = 1000;
a = 2;
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
        case 'tinitialization'
            t = parameterValue;
        case 'tmin'
            tmin = parameterValue;
        case 'tmax'
            tmax =  parameterValue;
        case 'sigma'
            sigma =  parameterValue;
        case 'eta'
            eta = parameterValue;
        case 'nonmonotone'
            M = parameterValue;

        otherwise
            error(['The parameter ''' parameterName ''' is not recognized by the function ''' mfilename '''!']);
    end
end
fprintf('OPE Logistic paras : a = %f, maxiter : %d, epsilon = %f\n', a, maxiter, epsilon);
w = w0; 
fun = zeros(maxiter+1,1);
time = fun;

logist = zeros(n,1);

% Initial function value
Z = sparse(1:n,1:n,y,n,n)*X; % = yx [n*d]

% todo - figure out how to calculate grad & fun
Zw = -Z*w; 
posind = (Zw > 0);
logist(posind) = 1 + exp(-Zw(posind));
logist(~posind) = 1 + exp(Zw(~posind));

temp = logist;
temp(posind) = 1./logist(posind);
temp(~posind) = (logist(~posind)-1)./logist(~posind);
grad =  -Z'*temp/n;

fun(1) = ( sum(log(logist(~posind))) + sum( Zw(posind)+log(logist(posind)) ) )/n ...
    + funRegC(w,d,lambda,theta,regtype);  % correct ok
fun_min = fun(1);
time(1) = 0;

for iter = 1:maxiter
    tic;
    
    w_old = w;

    % Tinh F'(w)
    Zw = -Z*w_old; 
    posind = (Zw > 0);
    logist(posind) = 1 + exp(-Zw(posind));
    logist(~posind) = 1 + exp(Zw(~posind));

    temp = logist;
    temp(posind) = 1./logist(posind);
    temp(~posind) = (logist(~posind)-1)./logist(~posind);
    grad =  -Z'*temp/n;

    dF = grad + derRegC(w_old, d, lambda, theta, epsilon, regtype);
    
    s_t = findDirection(dF, d, a);
    
    % update w
    alpha_ = 2 / (iter + 2);
    w = w_old + (s_t - w_old)*alpha_;

    % calculate fun(iter+1)
    Zw = -Z*w; 
    posind = (Zw > 0);
    logist(posind) = 1 + exp(-Zw(posind));
    logist(~posind) = 1 + exp(Zw(~posind));
    fun(iter+1) = (sum(log(logist(~posind))) + sum(Zw(posind) + log(logist(posind))))/n ...
        + funRegC(w,d,lambda,theta,regtype);  
    
    if(fun(iter+1) < fun_min)
        fun_min = fun(iter+1);
    end

    time(iter+1) = time(iter) + toc; 
end
