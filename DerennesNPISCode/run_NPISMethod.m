%This MATLAB script runs the Non-parametric Importance Sampling, Single Loop Method presented in Derennes et al.
%2018 for various analytical test cases. It was coded and adapted by Derek Manheim, 
%at the University of California, Irvine, Department of Civil & Environmental
%Engineering using MATLAB r2015b. It is a fully Global, Moment Independent,
%Monte Carlo (MC) based sampling method which relies on non-parametric
%importance sampling to improve the convergence of existing single loop
%approaches. The user must specify what test case to run (testfunc, options are 1,2,3),
%the number of MC sampling points(Nevalpts), as well as the repetition number (Rep), which specifies the 
%location of the Sobol sequence used for QMC sampling. In addition to the files provided in this download folder,
%in order to run this code users must have the statistics and machine learning toolbox package installed
%and should be running MATLAB r2015 or later. The results are automatically
%saved after running this script. This code incorporates a
%much faster and efficient implementation using both an efficient 2D kde estimation (@kde) and inverse gauss transform
%(IFGT) to estimate the bivariate densities. Spline interpolations are used
%to promote accuracy and facilitate convergence of this NPIS method. 
%Final Version, updated 5-8-2019 
clear all; close all; clc

%shuffle random numbers
rng shuffle

%Run specifications 
Rep = 1; %Location in the sobol sequence (1,2,3,4) to set up different repetitions
testfunc = 1; %Which analytical test cases?
Nevalpts = 256; %How many QMC sampling points? should be a factor of 2 
Delay=500+((Rep-1)*Nevalpts);

%%Set up variables to evaluate each test case
if testfunc == 1 %linear test function
   Nvar = 6;
   sigmas = [1,1,1,1,1,1];
   f = @(Xs) (1.5.*Xs(:,1)+1.6.*(Xs(:,2))+1.7.*(Xs(:,3))+1.8.*(Xs(:,4))+1.9.*(Xs(:,5))+2.*(Xs(:,6)));
   X = rand(Nevalpts, Nvar);
   for j = 1:Nvar
       Xs(:,j) = sqrt(2)*erfinv(2.*X(:,j)-1);
   end
   [Y] = f(Xs); 
elseif testfunc == 2 %non-linear test function (right skewed)
    Nvar = 6;
    sigmas = [1,1,1,1,1,1];
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
    sigmas = [1,1,1];
    Nvar = 3;
    lb1 = repmat([-pi -pi -pi],[Nevalpts 1]);
    ub1 = repmat([pi pi pi],[Nevalpts 1]); 
    %Parameters
    a = 7; b = 0.1;
    f = @(x) (sin(x(:,1))+ a.*(sin(x(:,2)).^2)+(b.*(x(:,3).^4).*sin(x(:,1))));
    X = rand(Nevalpts, Nvar);
    Xss = lb1+((ub1-lb1).*X);
    %Evaluate model
    [Y] = f(Xss);
    %Std normal space for Xs
    for j = 1:Nvar
        Xs(:,j) = sqrt(2)*erfinv(2.*X(:,j)-1);
    end
end

tic
%Perform initial KDE estimation
%convert X to pdf values
ms = zeros(1,Nvar);
for j = 1:Nvar
    FXpdf(:,j) = normpdf(Xs(:,j),ms(1,j),sigmas(1,j));
end

%convert y to pdf values
[bandwidth1,density1,xmesh1,~] = kdeBotev(Y,2^14);
FYpdf = interp1(xmesh1,density1,Y,'spline');

%compute joint pdf density using 2D KDE from MATLAB, apply 2D interpolation
bandwidth2 = cell(Nvar,1);
xmesh2 = cell(Nvar,1);
ymesh2 = cell(Nvar,1);
density2 = cell(Nvar,1);
 parfor j = 1:Nvar
    %Bi-variate KDE estimate using MATLAB's built in function
    [bandwidth2{j,1},density22,xmesh22,ymesh22] = kde2d([Xs(:,j),Y],2^10);
    density2{j,1} = density22; xmesh2{j,1} = xmesh22; ymesh2{j,1} = ymesh22;
    FXY(:,j) = interp2(xmesh2{j,1},ymesh2{j,1},density2{j,1},Xs(:,j),Y,'spline');
 end
 
 %Begin here
 %Distribution 2, g2 other Fxi(x) * Fy(y) estimator
 Np = Nevalpts*4; %keep similar ratio to 5000:20000
 tic
 z = randsample(Nevalpts,Np,true);
 parfor j = 1:Nvar
    covm = [sigmas(1,j)^2 0;0 bandwidth1^2];
    U = chol(covm);
    Ysm = Y(z,:);
    mu = [zeros(Np,1),Ysm]';
    C = normrnd(0,1,2,Np);
    g = U*C+mu;
    g = g';
    Xg(:,j) = g(:,1);
    Yg(:,j) = g(:,2);
 end
 
 %Compute the weights here
 for j = 1:Nvar
    FXpdf2(:,j) = normpdf(Xg(:,j),ms(1,j),sigmas(1,j));
 end
  parfor j = 1:Nvar
     %Compute estimates of KDE density for new samples, 1D and 2D
     [Ys(:,j)] = interp1(xmesh1(2:end),density1(2:end),Yg(:,j),'spline');
     [Ys2(:,j)] = interp2(xmesh2{j,1},ymesh2{j,1},density2{j,1},Xg(:,j),Yg(:,j),'spline');
     %pdf values 
     gpdf(:,j) = FXpdf2(:,j).*Ys(:,j);
     w(:,j) = abs(((FXpdf2(:,j).*Ys(:,j))-Ys2(:,j)))./gpdf(:,j);
  end
 
 %double check weights are positive
 w = abs(w);
  
 %mean of weights
 wt = mean(w,1);
 
 %gopt estimation
 bandwidthw = cell(Nvar,1);
 tic
 parfor j = 1:Nvar
    %Weighted Bi-variate KDE estimate using new KDE function to
    %obtain bandwidth and Inverse Gauss Transform to estimate density
    p = kde([Xg(:,j),Yg(:,j)]','rot',w(:,j)','Gaussian');
    [density3(:,j),b] = evalIFGT(p,[Xg(:,j),Yg(:,j)]',25);
    p2 = getBW(p); bandwidthw{j,1} = [p2(1,1),p2(2,1)];
 end
 toc
 
 %Now sample from gopt
  Npp = Np;
  %a) Kernel estimator, calculate U values
  parfor j = 1:Nvar
      z = randsample(Np,Np,true,w(:,j)); %weighted random sampling
      mu = [Xg(z,j),Yg(z,j)]; %means of 2D KDE
      covn = diag(bandwidthw{j,1}.^2); %covariance matrix
      g = mvnrnd(mu,covn);
      Xop(:,j) = g(:,1);
      Yop(:,j) = g(:,2);
      %Scattered interpolation to find pdf
      F = scatteredInterpolant(Xg(:,j),Yg(:,j),density3(:,j),'nearest');
      gpdf2(:,j) = F(Xop(:,j),Yop(:,j));
  end

 %Calculate delta indices
 for j = 1:Nvar
    FXpdf2f(:,j) = normpdf(Xop(:,j),ms(1,j),sigmas(1,j));
 end
 parfor j = 1:Nvar
     %Compute estimates of KDE density for new samples, 1D and 2D
     [Ysf(:,j)] = interp1(xmesh1(2:end),density1(2:end),Yop(:,j),'spline');
     [Ys2f(:,j)] = interp2(xmesh2{j,1},ymesh2{j,1},density2{j,1},Xop(:,j),Yop(:,j),'spline');
     deltast(:,j) = abs(((FXpdf2f(:,j).*Ysf(:,j))-Ys2f(:,j)))./gpdf2(:,j);
 end
 
 %calculate delta values
 delta = 0.5.*mean(deltast,1);
 toc
 
 %save results
 filename = strcat(['NPISSingleLoop_Test',num2str(testfunc),'_N',num2str(Nevalpts),'_Rep',num2str(Rep),'.mat']);
 save(filename,'delta','deltast','Xg','Yg','Xop','Yop','Y','X','Xs');
