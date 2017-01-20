% Before you run this example, make sure that you have run install.m
% to add the path and mex C files
clear;
clc;
close all;
% load data
filename = 'classic_binary.mat';
filepath = strcat('../data/',filename); 
Data = load (filepath);

y = Data.L';
X = Data.X';

clear Data

% [X,y] = readData('../../datasets/news20.binary');
[X,y] = readData('../../datasets/real-sim');

% statistics of the data
[n,d] = size(X);

% input parameters
% lambda = 1e-4;
% theta = 0.1 * lambda;
% optional parameter settings

regtype = 1; % nonconvex regularization type (default: 1 [capped L1]) 
lambda = 1e-4;
theta = 1e-1*lambda;

% regtype = 3;
% chay lau, ra len xuong neu lambda nho 1e-4 : 1e-5


fprintf('==== logistic loss ======\n');
fprintf('Data : no of instance = %d, Dimension = %d \n', n, d);
fprintf('Lambda = %f, theta = %f \n', lambda, theta);
fprintf('reg type = %d \n\n', regtype);

w0 = zeros(d,1) ; % starting point (default: zero vector)

stopcriterion = 1; % stopping criterion (default: 1)

maxiter = 1000; % number of maximum iteration (default: 1000)

tol = 1e-5; % stopping tolerance (default: 1e-5)

M = 5; % nonmonotone steps (default: 5)

t = 1; % initialization of t (default: 1)

tmin = 1e-30; % tmin parameter (default: 1e-20)

tmax = 1e30; % tmax parameter (default: 1e20)

sigma = 1e-5; % parameter in the line search (default: 1e-5)

eta = 2; % eta factor (default: 2)

stopnum = 3; % number of satisfying stopping criterion (default: 3)

maxinneriter = 20; % number of maximum inner iteration (line search) (default: 20)

% call the function
[w,fun,time,iter,fun_min] = gistLogistic(X,y,lambda,theta,...
                                  'maxiteration',maxiter,...
                                  'regtype',regtype,...
                                  'stopcriterion', stopcriterion,...
                                  'tolerance',tol,...
                                  'startingpoint',w0,...
                                  'nonmonotone',M,...
                                  'tinitialization',t,...
                                  'tmin',tmin,...
                                  'tmax',tmax,...
                                  'sigma',sigma,...
                                  'eta',eta,...
                                  'stopnum',stopnum,...
                                  'maxinneriter',maxinneriter);
fprintf('GIST: fun_min = %f \n', fun_min);
parse_count = 0;
for i = 1 : d 
  if w(i) == 0
      parse_count = parse_count + 1;
  end
end
fprintf('w : no of zero elements  = %d \n', parse_count);
fprintf('End of GIST \n\n\n');

                              
% plot
figure
% subplot(1,2,1);
semilogy(time(1:iter+1), fun(1:iter+1),'r-','LineWidth', 2);
xlabel('Iteration');
ylabel('Objective function value (log scaled)');
legend('GIST-Logistic')
hold on;

[w1,fun1,time1,iter1,fun_min1] = opeLogistic(...
      X,y,lambda,theta, ...
      'regtype',regtype, ...
      'startingpoint',w0, ...
      'maxiteration',50, ...
      'bound',1e2...
      );

fprintf('OPE: fun_min = %f \n', fun_min1);
parse_count = 0;
for i = 1 : d 
  if w1(i) == 0
      parse_count = parse_count + 1;
  end
end
fprintf('w : no of zero elements  = %d \n', parse_count);
fprintf('End of OPE \n\n\n');

% figure
% subplot(1,2,2);
semilogy(fun1(1:iter1+1),'b-','LineWidth', 2)
xlabel('Iteration');
ylabel('Objective function value (log scaled)');
legend('OPE-Logistic')
% % legend({['GIST-L2SVM'],'OPE-L2SVM'});
