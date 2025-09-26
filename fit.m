function [model, upper] = fit(X, y, param)

y_tilde=y;

var = [];
model = {};

n_train=length(y);

options = optimoptions('linprog', 'ConstraintTolerance', 1e-3, ...
    'OptimalityTolerance', 1e-3, 'Display', 'none');

for i=1:param.T

    %%% Train

    [var, y_tilde, mu, R, clf] = iboost(i, var, X, y, y_tilde, param, options);

    %%% Upper bound and classification error

    upper(i) = R;
    model{i} = clf;

end
model{end+1} = mu;
end

function [var, y_tilde, mu, upper, clf] = iboost(i, var, X, y, y_tilde, hyper_param, options)
%%%% One boost

[var, clf] = expectation_estimate(i, var, X, y, y_tilde);

c=[-var.tau+var.lambda;var.tau+var.lambda];
M=[var.M_train,-var.M_train;-var.M_train,var.M_train;-eye(i),zeros(i,i);zeros(i,i),-eye(i)];

n_train = length(y);

%%%%% Mosek
if hyper_param.solver == "Mosek"

    if i == 1
        var.res = [];
    end

    [sol, dual_sol, upper, var.res] = solver_mosek(i, n_train, c, M, var.res);
    %%% Linprog
elseif hyper_param.solver == "linprog"

    [sol, dual_sol, upper] = solver_linprog(i, n_train, c, M, options);

end
 % Update weights
alpha=dual_sol(1:n_train);
beta=dual_sol(n_train+1:2*n_train);
p = y/n_train-alpha+beta;
y_tilde=sign(p);
var.weights = abs(p);

mu=sol(1:i)-sol(i+1:2*i);

end

function [var, clf] = expectation_estimate(i, var, x_train, y, y_tilde)
%%% Uncertainty set
n_train = length(y);

if i == 1
    var.tau = [];
    var.lambda = [];
    var.M_train = [];
    % Initialize sample weights
    var.weights=ones(n_train,1)/n_train;
end

% Train weak learner
rng(i)
idx=randsample(n_train,n_train,true,var.weights);
clf=fitctree(x_train(:,idx)',y_tilde(idx),'MaxNumSplits',10,'NumVariablesToSample','all','OptimizeHyperparameters','none','HyperparameterOptimizationOptions', struct('UseParallel', true, 'ShowPlots', false));

% Make predictions
pred = predict(clf,x_train');
var.M_train=[var.M_train,pred];

var.tau=[var.tau;mean(y.*pred)];
var.lambda=[var.lambda; 1/sqrt(n_train)];

end

function [sol, dual_sol, upper, res] = solver_mosek(i, n_train, c, M, res)
%%% Optimization with Mosek
blc   = repmat(-inf, 2*n_train+2*i, 1);
prob.c = c;
S = sparse(M);
[i1,j1,s1] = find(S);
subi   = i1;
subj   = j1;
valij  = s1;
prob.a = sparse(subi, subj, valij);
prob.blc = blc;
buc = [0.5*ones(2*n_train,1);zeros(2*i,1)];
prob.buc = buc;
prob.blx = zeros(2*i, 1);
prob.bux = repmat(inf, 2*i, 1);
param = [];

%%%% Tolerance
param.MSK_DPAR_INTPNT_TOL_DFEAS = 1.0e-3;
param.MSK_DPAR_DATA_TOL_X = 1.0e-3;

%%%% Warm-start
if i > 1
    bas.xx = [res.sol.bas.xx; 0; 0];
    bas.slx = [res.sol.bas.slx;0;0];
    bas.sux = [res.sol.bas.sux;0;0];
    prob.sol.bas = bas;
end

[~,res] = mosekopt('minimize echo(0)',prob, param);
sol = res.sol.itr.xx;
dual_sol = res.sol.itr.y;
upper = res.sol.itr.pobjval+1/2;

end

function [sol, dual_sol, upper] = solver_linprog(i, n_train, c, M, options)
%%% Optimization with linprog
[sol,upper,~,~,multi]=linprog(c,M,[0.5*ones(2*n_train,1);zeros(2*i,1)],[],[],[],[],options);
dual_sol = multi.ineqlin;
upper = upper + 1/2;
end
