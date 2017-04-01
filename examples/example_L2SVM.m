clear;
clc;
close all;

% load data
Data = load ('../data/classic_binary.mat');
disp(Data);
y = Data.L';
X = Data.X';
clear Data

% statistics of the data
[n,d] = size(X);

% input parameters
lambda = 1e-3*abs(randn);
theta = inf;

% optional parameter settings
regtype = 1; % nonconvex regularization type (default: 1 [capped L1])
w0 = randn(d,1); % starting point (default: zero vector)
stopcriterion = 0; % stopping criterion (default: 1)
maxiter = 1000; % number of maximum iteration (default: 1000)
tol = 1e-5; % stopping tolerance (default: 1e-5)
M = 5; % nonmonotone steps (default: 5)
t = 1; % initialization of t (default: 1)
tmin = 1e-20; % tmin parameter (default: 1e-20)
tmax = 1e20; % tmax parameter (default: 1e20)
sigma = 1e-5; % parameter in the line search (default: 1e-5)
eta = 2; % eta factor (default: 2)
stopnum = 3; % number of satisfying stopping criterion (default: 3)
maxinneriter = 20; % number of maximum inner iteration (line search) (default: 20)


% call the function
[w,fun,time,iter] = gistL2SVM(X,y,lambda,theta,...
    'regtype',regtype, 'startingpoint', w0,...
    'stopcriterion', stopcriterion, 'tolerance',tol,...
    'nonmonotone',M, 'tinitialization',t, 'tmin',tmin, 'tmax',tmax,...
    'sigma',sigma, 'eta', eta, 'stopnum',stopnum,...
    'maxiteration', maxiter,'maxinneriter',maxinneriter);
% plot
figure
semilogy(time(1:iter+1),fun(1:iter+1),'r-','LineWidth', 2)
xlabel('CPU time (seconds)')
ylabel('Objective function value (log scaled)')
legend('GIST-L2SVM')
