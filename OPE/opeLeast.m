function [w,fun,time,iter,fun_min] = opeLeast(X,y,lambda,theta,varargin)


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

    fprintf('enter OPE Least %s \n', datestr(datetime));    

    [n,d] = size(X);
    w0 = zeros(d,1);
    maxiter = 100;
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
            otherwise
                error(['The parameter ''' parameterName ''' is not recognized by the function ''' mfilename '''!']);
        end
    end


    s_t = zeros(d,1);
    a = min([lambda, theta, lambda*theta]);
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

        dF = L(1) * (X'*(X*w_old - y))/n;

        for i = 1 : d
            switch regtype
                case 1 
                    if(w_old(i) > 0 && w_old(i) < theta)
                        dF(i) = dF(i) + L(2);
                    elseif (w_old(i) < 0 && w_old(i) > -theta)
                        dF(i) = dF(i) - L(2);
                    end            
                case 2 
                    if(w_old(i) > 0)
                        dF(i) = dF(i) + L(2) * lambda / (w_old(i) + theta);
                    elseif (w_old(i) < 0)
                        dF(i) = dF(i) + L(2) * lambda / (w_old(i) - theta);
                    else
                        tmp = randi([1,2],1);
                        if(tmp == 1)
                            dF(i) = dF(i) + L(2) * lambda / theta;
                        else
                            dF(i) = dF(i) - L(2) * lambda / theta;
                        end
                    end
                case 3 
                    if(w_old(i) > lambda && w_old(i) < theta * lambda)
                        dF(i) = dF(i) + L(2) * (-0.5*w_old(i) + 2 * lambda * theta) / (2 * theta - 2);
                    end
                    if(w_old(i) < -lambda && w_old(i) > -theta * lambda)
                        dF(i) = dF(i) + L(2) * (-0.5*w_old(i) - 2 * lambda * theta) / (2 * theta - 2);
                    end
                    if(w_old(i) > 0 && w_old(i) < lambda)
                        dF(i) = dF(i) + L(2) * lambda;
                    end
                    if(w_old(i) < 0 && w_old(i) > -lambda)
                        dF(i) = dF(i) + L(2) * (-lambda);
                    end
                case 4 
                    if (w_old(i) > 0 && w_old(i) < theta*lambda)
                       dF(i) = dF(i) + L(2) * (lambda - w_old(i)/theta);    
                    end
                    if (w_old(i) < 0 && w_old(i) > -theta*lambda)
                       dF(i) = dF(i) + L(2) * (-lambda - w_old(i)/theta);    
                    end
            end
        end


        % Tinh s_t = argmin<F'(w_old),x> x_i thuoc [-a, a] >
        [min_value, min_index] = min(dF);
        if min_value < 0
            for i = 1: d
                if i == min_index
                    s_t(i) = a;
                else 
                    s_t(i) = 0;
                end
            end
        else
            [max_value,max_index] = max(dF);
            for i = 1: d
                if i == max_index
                    s_t(i) = -a;
                else 
                    s_t(i) = 0;
                end
            end
        end
        %for i = 1 : d
            %if dF(i) > 0
             %   s_t(i) =  - a;
            % elseif dF(i) < 0
             %   s_t(i) = a;
           % end
        %end    

        w = w_old + (s_t - w_old) / iter;
        fun(iter+1) = 0.5*norm(X*w - y)^2/n + funRegC(w,d,lambda,theta,regtype);

        time(iter+1) = time(iter) + toc;
        if (fun(iter+1) < fun_min)
            fun_min = fun(iter+1);
        end
        
   
    end 



    fun = fun(1: min(maxiter,iter)+1);
    time = time(1: min(maxiter,iter)+1);
