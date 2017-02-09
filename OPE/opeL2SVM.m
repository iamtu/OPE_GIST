function [w,fun,time,iter,fun_min] = opeL2SVM(X,y,lambda,theta,varargin)

% OPE to solve L2 SVM loss
% Non-convex optimization problem:
%
% min_w L(w) + \sum_i r_i(w)
%
% ============================ loss function ==============================
%
% L(w) = 1/2n \sum_j \max(0,1 -y_j*x_j'*w)^2 (n: number of samples)
%
% ============================ regularizer ================================
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
%
% 'startingpoint': starting point (default: zero vector)
%
% 'maxiteration': number of maximum iteration (default: 100)
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
% fun_min : the minimum of f OPE can reach

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
maxiter = 100;
a = 1;
eps = 1e-6;
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
        case 'bound'
            a = parameterValue;
        case 'maxiteration'
            maxiter = parameterValue;
        case 'eps'
            eps = parameterValue;
        otherwise
            error(['The parameter ''' parameterName ''' is not recognized by the function ''' mfilename '''!']);
    end
end

fprintf('OPE params : a = %f, maxiter = %d, eps = %f\n',a, maxiter, eps);

w = w0; 
fun = zeros(maxiter+1,1); 
time = fun;
fun_min = fun(1);
% Initial function value
Z = sparse(1:n,1:n,y,n,n)*X; % Z = n x d
Zw = -Z*w; % Zw = n x 1
hinge = max(0,1+Zw); % hinge = n x 1
grad =  -Z'*hinge/n; % grad = d x 1

fun(1) = 0.5*(hinge'*hinge)/n + funRegC(w,d,lambda,theta,regtype);
time(1) = 0;
fun_min = fun(1);
L = [1,1];

for iter = 1:maxiter

    tic;
    
    w_old = w;
    
    % chon ngau nhien g1, g2
    randIndex = randi([1,2],1);
    L(randIndex) = L(randIndex) + 1;
    
    % Tinh F'(w)
    Zw = -Z*w_old; 
    hinge = max(0,1+Zw);
    grad =  -Z'*hinge/n;
    dF = L(1) * grad + L(2) * derRegC(w_old,d,lambda,theta, regtype);


    % Tinh s_t = argmin<F'(w_old),x> : sum_i |x_i|  <= a
    [min_value, min_index] = min(dF);
    if min_value < 0
        s_t = zeros(d,1);
        s_t(min_index) = a;
    else
        [max_value,max_index] = max(dF);
        s_t = zeros(d,1);
        s_t(max_index) = -a;
    end


    % Tinh s_t = argmax<F'(w_old),x> : sum_i |x_i|  <= a
    % [max_value, max_index] = max(dF);
    % if max_value > 0
    %     s_t = zeros(d,1);
    %     s_t(max_index) = a;
    % else
    %     [min_value,min_index] = min(dF);
    %     s_t = zeros(d,1);
    %     s_t(min_index) = -a;
    % end

    % s_t = zeros(d,1);
    % s_t (dF < 0) = a;
    % s_t (dF > 0) = -a;

    % update w
    alpha = 1.0 / iter;
    w = w_old * (1-alpha) + (s_t - w_old) * alpha;
    
    %calculate fun(iter+1)
    Zw = -Z*w;
    hinge = max(0,1+Zw);
    fun(iter+1) = 0.5*(hinge'*hinge)/n + funRegC(w,d,lambda,theta,regtype);

    time(iter+1) = time(iter) + toc;
    if (fun(iter+1) < fun_min)
        fun_min = fun(iter+1);
    end
    
end 