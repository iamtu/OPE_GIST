clear;
clc;
mkdir bin
mex ./GIST/proximalRegC.c -outdir ./bin;
mex ./GIST/funRegC.c -outdir ./bin;
mex ./OPE/funElementReg.c -outdir ./bin;
mex ./OPE/derRegC.c -outdir ./bin;

currentpath = cd;
addpath(genpath([currentpath,'/bin']));
