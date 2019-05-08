%This MATLAB script runs the Single Loop Method presented in Wei et al.
%2013 for various analytical test cases. It was coded and adapted by Derek Manheim, 
%at the University of California, Irvine, Department of Civil & Environmental
%Engineering using MATLAB r2015b. It is a fully Global, Moment Independent,
%Monte Carlo (MC) based sampling method which relies on non-parametric density estimation.
%The user must specify what test case to run (testfunc, options are 1,2,3),
%the number of MC sampling points(Nevalpts), as well as the repetition number (Rep), which specifies the 
%location of the Sobol sequence used for QMC sampling. In addition to the files provided in this download folder,
%in order to run this code users must have the statistics and machine learning toolbox package installed
%and should be running MATLAB r2015 or later. The results are automatically
%saved after running this script. 
%Final Version, updated 5-8-2019 
clear all; close all; clc

%Run specifications 
Rep = 1; %Location in the sobol sequence (1,2,3,4) to set up different repetitions
testfunc = 3; %Which analytical test cases?
Nevalpts = 256; %How many MC evaluation points used to construct and sample the Copula model? should be a factor of 2
Delay=500+((Rep-1)*Nevalpts);

if testfunc == 1 %linear test function
   Nvar = 6;
   f = @(Xs) (1.5.*Xs(:,1)+1.6.*(Xs(:,2))+1.7.*(Xs(:,3))+1.8.*(Xs(:,4))+1.9.*(Xs(:,5))+2.*(Xs(:,6)));
   p = sobolset(Nvar,'skip',Delay);
   X = p(1:Nevalpts,:);
   for j = 1:Nvar
       Xs(:,j) = sqrt(2)*erfinv(2.*X(:,j)-1);
   end
   [Y] = f(Xs); 
elseif testfunc == 2 %non-linear test function (right skewed)
    Nvar = 6;
    bi = [4.5,2.5,1.5,0.5,0.3,0.1];
    f = @(x,bi) exp(sum(bi.*x,2))-prod((exp(bi)-1)./bi);
    p = sobolset(Nvar,'skip',Delay);
    X = p(1:Nevalpts,:);
    for j = 1:Nevalpts
        Y(j,1) = exp(sum(bi.*X(j,:),2))-prod((exp(bi)-1)./bi);
    end
    for j = 1:Nvar
       Xs(:,j) = sqrt(2)*erfinv(2.*X(:,j)-1);
    end
elseif testfunc == 3 %Ishigami test function
    Nvar = 3;
    lb1 = repmat([-pi -pi -pi],[Nevalpts 1]);
    ub1 = repmat([pi pi pi],[Nevalpts 1]); 
    %Parameters
    a = 7; b = 0.1;
    f = @(x) (sin(x(:,1))+ a.*(sin(x(:,2)).^2)+(b.*(x(:,3).^4).*sin(x(:,1))));
    p = sobolset(Nvar,'skip',Delay);
    X = p(1:Nevalpts,:);
    Xss = lb1+((ub1-lb1).*X);
    %Evaluate model
    [Y] = f(Xss);
    %Std normal space for Xs
    for j = 1:Nvar
        Xs(:,j) = sqrt(2)*erfinv(2.*X(:,j)-1);
    end
end

%Single Loop Method Begins Here
%convert X to pdf values
FXpdf = normpdf(Xs,0,1);
    
%convert y to epdf values
[bandwidth,density1,xmesh,cdf] = kdeBotev(Y,2^14);
FYpdf = interp1(xmesh,density1,Y);

    %compute joint pdf density using 2D KDE from Botev, apply 2D interpolation
parfor j = 1:Nvar
   [bandwidth,density2,XX,YY] = kde2d([Xs(:,j),Y],2^10);
   FXY(:,j) = interp2(XX,YY,density2,Xs(:,j),Y);
end

%compute delta sensitivity indices
parfor j = 1:Nvar
   delta(1,j) = 0.5*mean(abs(((FXpdf(:,j).*FYpdf)./FXY(:,j))-1),1);
end

%Save Results
filename = strcat(['NPSingleLoop_Test',num2str(testfunc),'_N',num2str(Nevalpts),'_Rep',num2str(Rep),'.mat']);
save(filename,'delta','FXY','FYpdf','FXpdf','Y');
