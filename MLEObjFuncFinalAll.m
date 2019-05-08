function [LogL] = MLEObjFuncFinalAll(alpha,x2)
%This is the main optimization code in which LSHADEEpSinNLS calls to
%determine the fitness of each parent/child in the current population
%vectorized to run faster
%unpack variables
ry = x2{1,1};
rx = x2{1,2};
model = x2{1,3};

Nvar = size(rx,2);
Nevalpts = size(rx,1);
alphar = repmat(alpha(1,2:end),[Nevalpts 1]);
ryr = repmat(ry,[1 Nvar]);

%MLE begin here
%Estimate Yi
Yi = ((1-alphar).*ryr)+(alphar.*rx);

%Empirical CDFs, Y output
for j = 1:Nvar
    [~,Ypdf1,xmesh,Ycdf1] = kdeBotev(Yi(:,j),2^10);
    FYr(:,j) = interp1(xmesh,Ycdf1,Yi(:,j));
    Ypdf(:,j) = interp1(xmesh,Ypdf1,Yi(:,j));
end
FX = normcdf(rx,0,1);
Fpdf = normpdf(rx,0,1);

%Check if approximation returns values over 1 or below 0
for j = 1:Nvar
   [idx3] = find((FYr(:,j) > 1)); %upper boundary, 1
   [idx4] = find((FYr(:,j) < 0)); %lower boundary, 0
   FYr(idx3,j) = 1-eps; %set close to boundaries
   FYr(idx4,j) = eps;
end

%Spearman rho calculation
u = cell(Nvar,1);
rs = cell(Nvar,1);
nus = zeros(Nvar,1);
parfor j = 1:Nvar
    u{j,1} = [FYr(:,j) FX(:,j)];
    %Construct Copula Spearman from linear correlation matrix
    %Which Copula?
    if model == 1
        rs{j,1} = copulafit('Clayton',u{j,1});
    elseif model == 2
        rs{j,1} = copulafit('Frank',u{j,1});
    elseif model == 3
        rs{j,1} = copulafit('Gumbel',u{j,1});
    elseif model == 4
        rs{j,1} = copulafit('Gaussian',u{j,1});
    elseif model == 5
        [r,nu] = copulafit('t',u{j,1},'Method','ApproximateML');
        nus(j,1) = nu;
        rs{j,1} = r;
    end
end

%Gaussian copula density
parfor j = 1:Nvar
    if model == 1
        yc1(:,j) = copulapdf('Clayton',u{j,1},rs{j,1});
    elseif model == 2
        yc1(:,j) = copulapdf('Frank',u{j,1},rs{j,1});
    elseif model == 3
        yc1(:,j) = copulapdf('Gumbel',u{j,1},rs{j,1});
    elseif model == 4
        yc1(:,j) = copulapdf('Gaussian',u{j,1},rs{j,1});
    elseif model == 5
        yc1(:,j) = copulapdf('t',u{j,1},rs{j,1},nus(j,1));
    end
end

%take out zero values from copula
for j = 1:Nvar
    [m,p]= find(isinf(log(yc1(:,j)))==1); 
    yc1(m,:) = [];
end

%Likelihood calculation
fX3 = Nevalpts*sum(log(1-alpha));
fX1 = sum(sum((log(yc1)),1),2);
fX2 = sum(sum(log(Fpdf)+log(Ypdf),1),2);
LogL = -(fX3+fX1+fX2);

end