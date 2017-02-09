clear
clc


currentpath = cd;

addpath(genpath([currentpath,'/GIST']));
addpath(genpath([currentpath,'/OPE']));

cd ./GIST

mex proximalRegC.c
mex funRegC.c
cd ../OPE
mex funElementReg.c
mex derRegC.c
cd ..

