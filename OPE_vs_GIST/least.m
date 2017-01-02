% Before you run this example, make sure that you have run install.m
% to add the path and mex C files

clear;
clc;
close all;


% load data 
% Data = load ('../data/classic_binary.mat');

% disp(Data);

%y = Data.L';
% X = Data.X';

%clear Data

[X,y] = readData('../../datasets/web/w8a.txt');


[n,d] = size(X);
fprintf('no of instance = %d, dimension = %d\n', n, d);
wt = randn(d,1) *2 - 1;

% input parameters
lambda = 1e-3*abs(randn);
theta = 1e-2*lambda*abs(randn);
%theta = 3;
% optional parameter settings

regtype = 3; % nonconvex regularization type (default: 1 [capped L1]) 
fprintf('reg type = %d\n\n', regtype);

w0 = randn(d,1); % starting point (default: zero vector)

stopcriterion = 1; % stoppng criterion (default: 1)

maxiter = 100; % number of maximum iteration (default: 1000)

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
fprintf('Calling gist least...\n');

[w,fun,time,iter,fun_min] = gistLeast(X,y,lambda,theta,...
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
fprintf('iter = %d --- funtion = %f \n', iter+1, fun(iter+1));

parse_count = 0;

for i = 1 : d 
if w(i) == 0
    parse_count = parse_count + 1;
end
end
fprintf('parse count %d \n', parse_count);
fprintf('End of GIST \n\n\n');


% plot
figure
semilogy(time(1:iter+1),fun(1:iter+1),'r-','LineWidth', 2)
xlabel('CPU time (seconds)')
ylabel('Objective function value (log scaled)')
legend('GIST-Least')
hold on 
str = sprintf('  minvalue = %f', fun_min);
semilogy(0,fun_min,'k*', 'MarkerSize',6)
text(0,fun_min,str)
hold off


[w1,fun1,time1,iter1,fun_min1] = opeLeast(...
      X,y,lambda,theta,...
      'regtype',regtype,...
      'startingpoint',w0,...
      'maxiteration',500 ...
      );

fprintf('funtion = %f \n', fun1(iter1+1));
parse_count = 0;

for i = 1 : d 
if w1(i) == 0
    parse_count = parse_count + 1;
end
end
fprintf('parse count %d \n', parse_count);
fprintf('End of OPE \n');

figure
semilogy(time1(1:iter1+1),fun1(1:iter1+1),'b-','LineWidth', 2)
xlabel('CPU time (seconds)')
ylabel('Objective function value (log scaled)')
legend('OPE-Least')
hold on 
semilogy(0,fun_min1,'k*', 'MarkerSize',6)
str = sprintf('  minvalue %f', fun_min1);
semilogy(0,fun_min1,'k*', 'MarkerSize',6)
text(0,fun_min,str)
hold off

