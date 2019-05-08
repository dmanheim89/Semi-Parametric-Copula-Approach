%%%%%%%%%%%%%%%%%%%
%% This package is a MATLAB/Octave source code of LSHADE_EpSin which is an improved version of L-SHADE.
%% Please see the following paper:
%% * Noor H. Awad, Mostafa Z. Ali, Ponnuthurai N. Suganthan and Robert G. Reynolds: An Ensemble Sinusoidal Parameter Adaptation incorporated with L-SHADE for Solving CEC2014 Benchmark Problems, Proc. IEEE Congress on Evolutionary Computation (CEC-2016), Canada, July, 2016 
%% About L-SHADE, please see following papers:
%% Ryoji Tanabe and Alex Fukunaga: Improving the Search Performance of SHADE Using Linear Population Size Reduction,  Proc. IEEE Congress on Evolutionary Computation (CEC-2014), Beijing, July, 2014.
%%  J. Zhang, A.C. Sanderson: JADE: Adaptive differential evolution with optional external archive,” IEEE Trans Evol Comput, vol. 13, no. 5, pp. 945–958, 2009
%***Note this version does not use the LS (local search) method, hence "NLS"

function [solset] = LSHADEpSinNLS(fhd, d, LB, UB, N, tol, maxGen,x2)

rng(rand()*100);

%Specifically added for OED
% exptest = x2{1,2};

format long;
format compact;

problem_size = d;

%%% change freq
freq_inti = 0.5;
    
max_nfes = maxGen;

max_region = repmat(UB,[N 1]);
min_region = repmat(LB,[N 1]);
lu = [LB .* ones(1, problem_size); UB .* ones(1, problem_size)];

pb = 0.4;
ps = 0.5;

di = ceil(max_nfes/N);
S.Ndim = problem_size;
% S.Lband = ones(1, S.Ndim)*(min_region);
% S.Uband = ones(1, S.Ndim)*(max_region);
solset.Fbest = zeros(di,1);
solset.Xbest = zeros(di,d);
solset.Flist = zeros(di,N);
solset.Xlist= cell(di,d);

GenMaxSelected = 250; %%% For local search

%%%% Count the number of maximum generations before as NP is dynamically
%%%% decreased 
G_Max = 2163;
if problem_size == 10
    G_Max = 2163;
end
if problem_size == 30
    G_Max = 2745;
end
if problem_size == 50
    G_Max = 3022;
end
if problem_size == 100
    G_Max = 3401;
end

num_prbs = 1;
runs = 1;
run_funcvals = [];
RecordFEsFactor = ...
	[0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, ...
	0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
progress = numel(RecordFEsFactor);

allerrorvals = zeros(progress, runs, num_prbs);
result=zeros(num_prbs,5);

fprintf('Running LSHADE_EpSin algorithm on D= %d\n', problem_size) 
% for func = 1 : num_prbs
%   optimum = func * 100.0;
%   S.FuncNo = func;
  
  %% Record the best results
  outcome = []; 

%   fprintf('\n-------------------------------------------------------\n')
%   fprintf('Function = %d, Dimension size = %d\n', func, problem_size) 

  for run_id = 1 : runs
      
     run_funcvals = [];
     col=1;              %% to print in the first column in all_results.mat
     
    %%  parameter settings for L-SHADE
    p_best_rate = 0.11;    %0.11
    arc_rate = 1.4;
    memory_size = 5;
%     pop_size = 18 * problem_size;   %18*D
    pop_size = N;
    SEL = round(ps*pop_size);

    max_pop_size = pop_size;
    min_pop_size = 4.0;

     nfes = 0;
  end

    %Set up initial population and evaluate fitnesses of parents here
    popold = repmat(lu(1, :), pop_size, 1) + rand(pop_size, problem_size) .* (repmat(lu(2, :) - lu(1, :), pop_size, 1));
    pop = popold; % the old population becomes the current population
    parfor i = 1:length(pop)
        [fitness(i,1)] = feval(fhd,pop(i,:),x2);
    end
    
    run_funcvals = [run_funcvals;fitness];
 
    bsf_fit_var = 1e+30;
    bsf_index = 0;
    bsf_solution = zeros(1, problem_size);
    
    %%%%%%%%%%%%%%%%%%%%%%%% for out
    for i = 1 : pop_size
        nfes = nfes + 1;
        
        if fitness(i) < bsf_fit_var
            bsf_fit_var = fitness(i);
            bsf_solution = pop(i, :);
            bsf_index = i;
        end
        
        if nfes > max_nfes; break; end
    end
    %%%%%%%%%%%%%%%%%%%%%%%% for out
    
    memory_sf = 0.5 .* ones(memory_size, 1);
    memory_cr = 0.5 .* ones(memory_size, 1);

    memory_freq = freq_inti*ones(memory_size, 1);
    memory_pos = 1;

    archive.NP = arc_rate * pop_size; % the maximum size of the archive
    archive.pop = zeros(0, problem_size); % the solutions stored in te archive
    archive.funvalues = zeros(0, 1); % the function value of the archived solutions

    %% main loop
    gg=0;  %%% generation counter used For Sin
    igen =1;  %%% generation counter used For LS
 
    flag1 = false;
    flag2 = false;
    while nfes < max_nfes
      gg=gg+1;
         
      pop = popold; % the old population becomes the current population
      [temp_fit, sorted_index] = sort(fitness, 'ascend');

      mem_rand_index = ceil(memory_size * rand(pop_size, 1));
      mu_sf = memory_sf(mem_rand_index);
      mu_cr = memory_cr(mem_rand_index);
      mu_freq = memory_freq(mem_rand_index);

      %% for generating crossover rate
      cr = normrnd(mu_cr, 0.1);
      term_pos = find(mu_cr == -1);
      cr(term_pos) = 0;
      cr = min(cr, 1);
      cr = max(cr, 0);
      
      %% for generating scaling factor
      sf = mu_sf + 0.1 * tan(pi * (rand(pop_size, 1) - 0.5));
      pos = find(sf <= 0);
      
      while ~ isempty(pos)
          sf(pos) = mu_sf(pos) + 0.1 * tan(pi * (rand(length(pos), 1) - 0.5));
          pos = find(sf <= 0);
      end
      
      freq = mu_freq + 0.1 * tan(pi*(rand(pop_size, 1) - 0.5));
      pos_f = find(freq <=0);
      while ~ isempty(pos_f)
        freq(pos_f) = mu_freq(pos_f) + 0.1 * tan(pi * (rand(length(pos_f), 1) - 0.5));
        pos_f = find(freq <= 0);
      end

      sf = min(sf, 1);
      freq = min(freq, 1);
      
      if(nfes <= max_nfes/2)
          c=rand;
          if(c<0.5)
              sf = 0.5.*( sin(2.*pi.*freq_inti.*gg+pi) .* ((G_Max-gg)/G_Max) + 1 ) .* ones(pop_size,problem_size);
          else
              sf = 0.5 *( sin(2*pi .* freq(:, ones(1, problem_size)) .* gg) .* (gg/G_Max) + 1 ) .* ones(pop_size,problem_size);
          end
      end
     
      r0 = [1 : pop_size];
      popAll = [pop; archive.pop];
      [r1, r2] = gnR1R2(pop_size, size(popAll, 1), r0);
      
      pNP = max(round(p_best_rate * pop_size), 2); %% choose at least two best solutions
      randindex = ceil(rand(1, pop_size) .* pNP); %% select from [1, 2, 3, ..., pNP]
      randindex = max(1, randindex); %% to avoid the problem that rand = 0 and thus ceil(rand) = 0
      pbest = pop(sorted_index(randindex), :); %% randomly choose one of the top 100p% solutions

      vi = pop + sf(:, ones(1, problem_size)) .* (pbest - pop + pop(r1, :) - popAll(r2, :));
      vi = boundConstraint(vi, pop, lu);
      
      mask = rand(pop_size, problem_size) > cr(:, ones(1, problem_size)); % mask is used to indicate which elements of ui comes from the parent
      rows = (1 : pop_size)'; cols = floor(rand(pop_size, 1) * problem_size)+1; % choose one position where the element of ui doesn't come from the parent
      jrand = sub2ind([pop_size problem_size], rows, cols); mask(jrand) = false;
      ui = vi; ui(mask) = pop(mask);
      
      %Check if parameters are in bounds, reflect if not
      [ui] = xintobounds(ui, LB, UB);
      
      %Evaluate Children fitnesses
       parfor i = 1:size(ui,1)
           [children_fitness(i,1)] = feval(fhd, ui(i,:),x2);
       end
%       children_fitness = feval(fhd, ui', func);
%       children_fitness = children_fitness';

     
      %%%% To check stagnation
      flag = false;
      bsf_fit_var_old = bsf_fit_var;
      %%%%%%%%%%%%%%%%%%%%%%%% for out
      for i = 1 : pop_size
          nfes = nfes + 1;
          
          if children_fitness(i) < bsf_fit_var
              bsf_fit_var = children_fitness(i);
              bsf_solution = ui(i, :);
              bsf_index = i; 
          end
          
          if nfes > max_nfes; break; end
      end      
      %%%%%%%%%%%%%%%%%%%%%%%% for out

      dif = abs(fitness - children_fitness);


      %% I == 1: the parent is better; I == 2: the offspring is better
      I = (fitness > children_fitness);
      goodCR = cr(I == 1);  
      goodF = sf(I == 1);
      goodFreq = freq(I == 1);
      dif_val = dif(I == 1);

%      isempty(popold(I == 1, :))   
      archive = updateArchive(archive, popold(I == 1, :), fitness(I == 1));

      [fitness, I] = min([fitness, children_fitness], [], 2);
      
      run_funcvals = [run_funcvals; fitness];
      
      popold = pop;
      popold(I == 2, :) = ui(I == 2, :);

      num_success_params = numel(goodCR);

      if num_success_params > 0
          sum_dif = sum(dif_val);
          dif_val = dif_val / sum_dif;
          
          %% for updating the memory of scaling factor
          memory_sf(memory_pos) = (dif_val' * (goodF .^ 2)) / (dif_val' * goodF);
          
          %% for updating the memory of crossover rate
          if max(goodCR) == 0 || memory_cr(memory_pos)  == -1
              memory_cr(memory_pos)  = -1;
          else
              memory_cr(memory_pos) = (dif_val' * (goodCR .^ 2)) / (dif_val' * goodCR);
          end
          
          %% for updating the memory of freq
          if max(goodFreq) == 0 || memory_freq(memory_pos)  == -1
              memory_freq(memory_pos)  = -1;
          else
              memory_freq(memory_pos) = (dif_val' * (goodFreq .^ 2)) / (dif_val' * goodFreq);
          end
          
          memory_pos = memory_pos + 1;
          if memory_pos > memory_size;  memory_pos = 1; end
      end

      %% for resizing the population size
      plan_pop_size = round((((min_pop_size - max_pop_size) / max_nfes) * nfes) + max_pop_size);

      if pop_size > plan_pop_size
          reduction_ind_num = pop_size - plan_pop_size;
          if pop_size - reduction_ind_num <  min_pop_size; reduction_ind_num = pop_size - min_pop_size;end
          
          pop_size = pop_size - reduction_ind_num;
          SEL = round(ps*pop_size);
          for r = 1 : reduction_ind_num
              [valBest indBest] = sort(fitness, 'ascend');
              worst_ind = indBest(end);
              popold(worst_ind,:) = [];
              pop(worst_ind,:) = [];
              fitness(worst_ind,:) = [];
          end
          
          archive.NP = round(arc_rate * pop_size);
          
          if size(archive.pop, 1) > archive.NP
              rndpos = randperm(size(archive.pop, 1));
              rndpos = rndpos(1 : archive.NP);
              archive.pop = archive.pop(rndpos, :);
          end
      end
      
        popsizestore(gg,1) = pop_size;
            
        %Termination criteria
        tol1 = tol;
        rev2 = range(fitness,1);
        children_fitness = [];
            if (rev2 <= tol1) 
                disp('Local Convergence Reached, LSHADE Stopped, tol1');
                disp( strcat( 'FunEvals# : ',num2str(sum(popsizestore)) ));
                disp( strcat( 'Generation# : ',num2str(gg) ));
                break;
            else
                [valBest, indBestn] = sort(fitness, 'ascend');
                solset.Fbest(gg,1) = fitness(indBestn(1,1));
                solset.Xbest(gg,1:d) = pop(indBestn(1,1),:);
                solset.Flist(gg,1:length(fitness)) = fitness';
                for hj = 1:d
                    solset.Xlist{gg,hj} = pop(:,hj)';
                end
                display(['Generation#:',num2str(gg)]);
                display(['Best:',num2str(solset.Fbest(gg,1))]);
                display(['Range:',num2str(rev2)]);
            end
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end %%%%%%%%nfes  
  end %% end 1 run
% end %% end 1 function run

