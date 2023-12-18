function logLik=GARCHlnL(theta, X)
% Output must be a column vector

% Extract parameters
w=theta(1);
a=theta(2);
b=theta(3);
mu=theta(4);

T = length(X);

% Initialize variables
logLik = 0;
sigma2 = zeros(T,1);
epsilon2 = zeros(T,1);

% Initialize first conditional variance (Stationary variance of GARCH(1,1))
sigma2(1) = w / (1 - a - b);

for t = 2:T
    % Squared residuals
    epsilon2(t) = (X(t) - mu)^2;

    % Conditional variance
    sigma2(t) = w + a*epsilon2(t-1) + b*sigma2(t-1);

    % Update logLik contribution from this specific datapoint
    logLik = logLik - 0.5*log(2*pi*sigma2(t)) - 0.5*epsilon2(t)/sigma2(t);
end


end