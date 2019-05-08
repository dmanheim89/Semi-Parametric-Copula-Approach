%This MATLAB script runs the Semi-Parametric Copula Method presented in Manheim &
%Detwiler (2019) for various analytical test cases. It was originally coded and developed by Derek Manheim, 
%at the University of California, Irvine, Department of Civil & Environmental
%Engineering using MATLAB r2015b. It is a fully Global, Moment Independent, Monte Carlo (MC) based sampling method
%that relies on the use of parametric copula models to describe the joint pdf between
%model inputs (i.e., parameters) and outputs. The user must specify what
%test case to run (testfunc, options are 1,2,3), the copula model used 
%(options are 'Gaussian','t','Clayton','Frank','Gumbel'), the number of MC sampling points used to construct 
%and sample the Copula model (Nevalpts) as well as the repetition number (Rep), which specifies the 
%location of the Sobol sequence used for QMC sampling. In addition to the files provided in this download folder,
%in order to run this code users must have the statistics and machine learning toolbox package installed
%and should be running MATLAB r2015 or later. The results are automatically
%saved after running this script. *Note - running over 65536 Nevalpts will
%require more advanced parallel computation (8-16 cores or more) and is not recommended to be run locally. 
%Final Version, updated 5-8-2019 
clear all; close all; clc

%Run specifications 
Rep = 1; %Location in the sobol sequence (1,2,3,4) to set up different repetitions
testfunc = 1; %Which analytical test cases?
CopulaModel = 'Frank'; %Which parametric Copula Model to use?
Nevalpts = 256; %How many MC evaluation points used to construct and sample the Copula model? should be a factor of 2
Delay=500+((Rep-1)*Nevalpts); %This sets the location in the sobol sequence for QMC sampling

%Set up variables required for MLE optimization
if strcmp(CopulaModel,'Clayton')
    model = 1;
elseif strcmp(CopulaModel,'Frank')
    model = 2;
elseif strcmp(CopulaModel,'Gumbel')
    model = 3;
elseif strcmp(CopulaModel,'Gaussian')
    model = 4;
elseif strcmp(CopulaModel,'t')
    model = 5;
end

%%Set up variables to evaluate each test case
if testfunc == 1 %linear test case
   Nvar = 6;
   f = @(Xs) (1.5.*Xs(:,1)+1.6.*(Xs(:,2))+1.7.*(Xs(:,3))+1.8.*(Xs(:,4))+1.9.*(Xs(:,5))+2.*(Xs(:,6)));
   p = sobolset(Nvar,'skip',Delay);
   X = p(1:Nevalpts,:);
   for j = 1:Nvar
       Xs(:,j) = sqrt(2)*erfinv(2.*X(:,j)-1); %Model inputs
   end
  [Yn] = f(Xs); 
   Y = Yn; %Model output
elseif testfunc == 2 %non-linear test case
    Nvar = 6;
    bi = [4.5,2.5,1.5,0.5,0.3,0.1];
    f = @(x,bi) exp(sum(bi.*x,2))-prod((exp(bi)-1)./bi);
    p = sobolset(Nvar,'skip',Delay);
    X = p(1:Nevalpts,:);
    for j = 1:Nevalpts
        Yn(j,1) = exp(sum(bi.*X(j,:),2))-prod((exp(bi)-1)./bi);
    end
    Y = Yn; %model outputs
    for j = 1:Nvar
       Xs(:,j) = sqrt(2)*erfinv(2.*X(:,j)-1); %model inputs
    end
elseif testfunc == 3 %Ishigami test case (non-linear/non-monotonic)
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
    [Yn] = f(Xss);
    Y = Yn; %model output
    %Std normal space for Xs
    for j = 1:Nvar
        Xs(:,j) = sqrt(2)*erfinv(2.*X(:,j)-1); %model inputs
    end
end

%%Estimate alpha values for the Rolling Pin method
%using the maximum likelihood (MLE) optimization approach
%Here, an external call to LSHADEEpSinNLS algorithm is performed
%Note, on 2 cores this will likely take a bit of time, especially if
%Nevalpts is high
ry = Y;
rx = Xs;
Ngens = 20000; %This is the number of generations used for MLE optimization
Npop = 50; %This is the number of population members used for MLE optimization
nTol = 1E-6; %This is the tolerance used to moniter convergence 
paramL = zeros(1,Nvar+1); paramU = ones(1,Nvar+1); %boundaries of the alpha parameter values, Rolling Pin method
paramU(1,1) = 0; 
x2{1,1} = ry; x2{1,2} = rx; x2{1,3} = model;
[solset] = LSHADEpSinNLS(@MLEObjFuncFinalAll, Nvar+1, paramL, paramU, Npop, nTol, Ngens,x2); %Run optimization call
[m,p] = find(nonzeros(solset.Xbest(:,2)));
alpha = solset.Xbest(m(end),:); %Extract best alpha values for Rolling Pin method

%%Semi-Parametric method begins here
%Estimate Yi, transformed values, Equation 4
for j = 1:Nvar
    Yi(:,j) = (1-alpha(1,j+1)).*(ry(:,1))+alpha(1,j+1).*(rx(:,j));
end

%Find the PDFs and CDFs of the transformed distributions, Yi and Xs (Eqs.
%5 and 6)
parfor j = 1:Nvar
   tic
   pd1{j,1} = fitdist(Yi(:,j),'Kernel');
   FYr(:,j) = cdf(pd1{j,1},Yi(:,j));
   FX(:,j) = normcdf(rx(:,j),0,1);
   toc
end

%Fit Elliptical or Archimidean parametric copula model here using MATLAB's built in
%functions
for j = 1:Nvar
   u{j,1} = [FYr(:,j) FX(:,j)];
   %Construct Parametric Copula from CDF matrix, t-distribution is a
   %special case
   if model == 5
       [r{j,1},nus{j,1}] = copulafit(CopulaModel,u{j,1});
   else
       [r{j,1}] = copulafit(CopulaModel,u{j,1});
   end
end

%Sample from constructed parametric copula model, using MATLAB's built in
%functions
parfor j = 1:Nvar
    tic
    if model == 5
        cs = copularnd(CopulaModel,r{j,1},nus{j,1},Nevalpts);
    else
        cs = copularnd(CopulaModel,r{j,1},Nevalpts);
    end
    FYrn(:,j) = cs(:,1);
    FXn(:,j) = cs(:,2);
    Yin(:,j) = icdf(pd1{j,1},FYrn(:,j));
    Xn(:,j) = norminv(FXn(:,j),0,1);
    toc
end

%Find the marginal PDFs of the sampled dists, Yin and Xn (Eqs. 5 and 6)
parfor j = 1:Nvar
   tic
   FYpdf1(:,j) = pdf(pd1{j,1},Yin(:,j));
   Fpdf(:,j) = normpdf(Xn(:,j),0,1);
   toc
end

%Transform the sample back to original space, inverse of Equation 4
parfor j = 1:Nvar
   Ys(:,j) = (Yin(:,j)-alpha(1,j+1).*Xn(:,j))./(1-alpha(1,j+1));
   pd3{j,1} = fitdist(Ys(:,j),'Kernel');
   FyPDF(:,j) = pdf(pd3{j,1},Ys(:,j));
end

%Evaluate the parametric copula density
for j = 1:Nvar
    %Evaluate joint pdf directly at pts 
    a = FYpdf1(:,j).*(1-alpha(1,j+1));
    b = normpdf(Xn(:,j),0,1).*(1-alpha(1,1));
    if model == 5
        fXa1(:,j) = copulapdf(CopulaModel,[FYrn(:,j),FXn(:,j)],r{j,1},nus{j,1});
    else
        fXa1(:,j) = copulapdf(CopulaModel,[FYrn(:,j),FXn(:,j)],r{j,1});
    end
    fX(:,j) = fXa1(:,j).*a.*b; %Equation 7
end

%Estimate the MI sensitivity indices, Equation 8
for j = 1:Nvar
    deltast(:,j) = abs(((FyPDF(:,j).*Fpdf(:,j))./(fX(:,j)))-1);
    delta(1,j) = 0.5*mean(deltast(:,j));
end

%Save results here
filename = strcat(['SemiParamCopula_',CopulaModel,'_Test',num2str(testfunc),'_N',num2str(Nevalpts),'_Rep',num2str(Rep),'.mat']);
save(filename,'delta','deltast','solset','Xs','Y','ry','rx');