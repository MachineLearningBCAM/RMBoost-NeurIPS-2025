function y = predict_boost(model, X)
T = length(model) - 1;
M = [];
for i = 1:T
    if ~isempty(model{i})
    M=[M,predict(model{i}, X')];
    end
end
y = sign(M*model{end});
end