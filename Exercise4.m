%% Project E4
%% Prep and check data
clc;
clear;
close all;

BTCtable = readtable('BTC-USD.csv');
NASDAQtable = readtable('NYA.csv');
LQDtable = readtable('LQD.csv');

% Find common dates
[Dates, IDates] = intersect(BTCtable.Date, NASDAQtable.Date);

Prices = [NASDAQtable.Close LQDtable.Close BTCtable.Close(IDates)];
Returns = log(Prices(2:end,:)) - log(Prices(1:end-1,:));
%Returns = Prices(2:end,:) - Prices(1:end-1,:);


figure(1);
Asset = {'NASDAQ', 'LQD', 'BTC'};
for d = 1:3
    subplot(3, 1, d)
    plot(NASDAQtable.Date(2:end), Returns(:,d))
    title(Asset{d})
end

%% Modelling 1
% muUncond = [mean(Returns(1:757,1)); mean(Returns(1:757,2)); mean(Returns(1:757,3))]
% cov12 = cov(Returns(1:757,1), Returns(1:757,2));
% cov13 = cov(Returns(1:757,1), Returns(1:757,3));
% cov23 = cov(Returns(1:757,2), Returns(1:757,3));
% varUncond = [var(Returns(1:757,1)), cov12(1,2), cov13(1,2);...
%     cov12(1,2), var(Returns(1:757,2)), cov23(1,2);...
%     cov13(1,2), cov23(1,2), var(Returns(1:757,3))]

muUncond = [mean(Returns(:,1)); mean(Returns(:,2)); mean(Returns(:,3))]
cov12 = cov(Returns(:,1), Returns(:,2));
cov13 = cov(Returns(:,1), Returns(:,3));
cov23 = cov(Returns(:,2), Returns(:,3));
varUncond = [var(Returns(:,1)), cov12(1,2), cov13(1,2);...
    cov12(1,2), var(Returns(:,2)), cov23(1,2);...
    cov13(1,2), cov23(1,2), var(Returns(:,3))]

a = (ones(1,3)/varUncond)*muUncond;
b = (muUncond'/varUncond)*muUncond;
c = (ones(1,3)/varUncond)*ones(3,1);

%a = (ones(1,3)*inv(varUncond))*muUncond;
%b = (muUncond'*inv(varUncond))*muUncond;
%c = (ones(1,3)*inv(varUncond))*ones(3,1);

%NASDAQCap = 337039282*14,339.99;
%LDQCap = 21624183*;
%BTCCap = 0;

% NASDAQCap = NASDAQtable.Volume(end)*NASDAQtable.Close(end);
% LQDCap = LQDtable.Volume(end)*LQDtable.Close(end);
% BTCCap = BTCtable.Volume(end)*BTCtable.Close(end);

% NASDAQCap = NASDAQtable.Volume(630)*NASDAQtable.Close(630);
% LQDCap = LQDtable.Volume(630)*LQDtable.Close(630);
% BTCCap = BTCtable.Volume(913)*BTCtable.Close(913);

NASDAQCap = NASDAQtable.Volume(1)*NASDAQtable.Close(1);
LQDCap = LQDtable.Volume(1)*LQDtable.Close(1);
BTCCap = BTCtable.Volume(1)*BTCtable.Close(1);

%markCaps = [31.34e9; 1.16e9*1.08; 861.123e9];    % Market capitalization for [NASDAQ, LQD, BTC-USD]
%markCaps = [31.34e9; 1.16e9*1.08; 25.415e9]; % Unsure if this BTC-USD value or the one above is the MC
%markCaps = [13.34e9; 1.16e9*1.08; 203.881e9];
markCaps = [NASDAQCap; LQDCap; BTCCap];

wMVP = markCaps./sum(markCaps)              % Weighted market caps
%wMVP = log(markCaps)./sum(log(markCaps))   % Log-weighted market caps
%wMVP = (varUncond\ones(3,1))*(1/c          % MVP portfolio weights
%wMVP = [1/3;1/3;1/3];                      % Equally dist. weights
%wMVP = [30; 40; 30]./100;                  % Custom weights

expRetMVP = muUncond'*wMVP

%expRetMVP = 0.00024516;
% 0.000211095013562748
%expRetMVP = -1.999788904986437;
%expRetMVP = 0.135363854855620

a = (ones(1,3)/varUncond)*muUncond;
%a = (ones(1,3)*inv(varUncond))*muUncond;
b = (muUncond'/varUncond)*muUncond;
%b = (muUncond'*inv(varUncond))*muUncond;


gamma = (b-(a^2)/c)/(expRetMVP - a/c)

%rf = log(0.05);
%rf = 0.05;

%gamma = 2*(expRetMVP - rf)/(wMVP'*varUncond*wMVP)

%% Est GARCH
retsTrain = Returns(1:757,:)';

negLogLikelihoodNASDAQ = @(params) -GARCHlnL(params, retsTrain(1,:));
negLogLikelihoodLQD = @(params) -GARCHlnL(params, retsTrain(2,:));
negLogLikelihoodBTC = @(params) -GARCHlnL(params, retsTrain(3,:));

% Initial guess for parameters [omega, alpha, beta, mu]
%initialParamsNASDAQ = [0.00001, 0.1, 0.8, mean(retsTrain(1,:))];
%initialParamsLQD = [0.00001, 0.1, 0.8, mean(retsTrain(2,:))];
%initialParamsBTC = [0.00001, 0.05, 0.9, mean(retsTrain(3,:))];
initialParamsNASDAQ = [5e-6, 0.24, 0.72, mean(retsTrain(1,:))];
initialParamsLQD = [2e-6, 0.23, 0.68, mean(retsTrain(2,:))];
initialParamsBTC = [0.0003, 0.14, 0.73, mean(retsTrain(3,:))];

% Use fminunc to maximize the log-likelihood
estimatedParamsNASDAQ = fminsearch(negLogLikelihoodNASDAQ, initialParamsNASDAQ);
estimatedParamsLQD = fminsearch(negLogLikelihoodLQD, initialParamsLQD);
estimatedParamsBTC = fminsearch(negLogLikelihoodBTC, initialParamsBTC);

format long
disp('Estimated Parameters for NASDAQ using fminsearch (omega, alpha, beta, mu):');
disp(estimatedParamsNASDAQ);
disp('Alpha + Beta')
disp(estimatedParamsNASDAQ(2) + estimatedParamsNASDAQ(3))
format short

format long
disp('Estimated Parameters for LQD using fminsearch (omega, alpha, beta, mu):');
disp(estimatedParamsLQD);
disp('Alpha + Beta')
disp(estimatedParamsLQD(2) + estimatedParamsLQD(3))
format short

format long
disp('Estimated Parameters for BTC using fminsearch (omega, alpha, beta, mu):');
disp(estimatedParamsBTC);
disp('Alpha + Beta')
disp(estimatedParamsBTC(2) + estimatedParamsBTC(3))
format short

N = length(retsTrain(1,:));

w1 = estimatedParamsNASDAQ(1);
a1 = estimatedParamsNASDAQ(2);
b1 = estimatedParamsNASDAQ(3);
mu1 = estimatedParamsNASDAQ(4);

w2 = estimatedParamsLQD(1);
a2 = estimatedParamsLQD(2);
b2 = estimatedParamsLQD(3);
mu2 = estimatedParamsLQD(4);

w3 = estimatedParamsBTC(1);
a3 = estimatedParamsBTC(2);
b3 = estimatedParamsBTC(3);
mu3 = estimatedParamsBTC(4);

% model = garch(1,1);
% [fit1, ~, logl1] = estimate(model, retsTrain(1,:)');
% [fit2, ~, logl2] = estimate(model, retsTrain(2,:)');
% [fit3, ~, logl3] = estimate(model, retsTrain(3,:)');
% 
% [fit1, ~, logl1] = estimate(model, (retsTrain(1,:) - mean(retsTrain(1,:)))');
% [fit2, ~, logl2] = estimate(model, (retsTrain(2,:) - mean(retsTrain(2,:)))');
% [fit3, ~, logl3] = estimate(model, (retsTrain(3,:) - mean(retsTrain(3,:)))');
% 
% w1 = fit1.Constant;
% a1 = fit1.ARCH{1};
% b1 = fit1.GARCH{1};
% %mu1 = fit1.Offset;
% mu1 = mean(retsTrain(1,:));
% 
% w2 = fit2.Constant;
% a2 = fit2.ARCH{1};
% b2 = fit2.GARCH{1};
% %mu2 = fit2.Offset;
% mu2 = mean(retsTrain(2,:));
% 
% w3 = fit3.Constant;
% a3 = fit3.ARCH{1};
% b3 = fit3.GARCH{1};
% %mu3 = fit3.Offset;
% mu3 = mean(retsTrain(3,:));

Pest = 0;
ZT = zeros(3,N);

sigma1 = zeros(N+1, 1);
sigma2 = zeros(N+1, 1);
sigma3 = zeros(N+1, 1);

sigma1(1) = w1 / (1 - a1 - b1);
sigma2(1) = w2 / (1 - a2 - b2);
sigma3(1) = w3 / (1 - a3 - b3);

returns = zeros(3,N);

for i = 1:N
    ZT(:,i) = (diag([1/sqrt(sigma1(i)), 1/sqrt(sigma2(i)), 1/sqrt(sigma3(i))])*...
        (retsTrain(:,i)-[mu1, mu2, mu3]'))';
    sigma1(i+1) = w1 + a1*(retsTrain(1,i)-mu1)^2 + b1*sigma1(i);
    sigma2(i+1) = w2 + a2*(retsTrain(2,i)-mu2)^2 + b2*sigma2(i);
    sigma3(i+1) = w3 + a3*(retsTrain(3,i)-mu3)^2 + b3*sigma3(i);
%     returns(:,i) = [sqrt(sigma1(i))*ZT(1,i); sqrt(sigma2(i))*ZT(2,i);
%         sqrt(sigma3(i))*ZT(3,i)];
    Pest = Pest + ZT(:,i)*ZT(:,i)';
end

for i = 1:N
    returns(:,i) = diag([sqrt(sigma1(i)), sqrt(sigma2(i)), sqrt(sigma3(i))])*randn(3,1);
end

Pest = Pest/N

rho12 = Pest(1,2);
rho13 = Pest(1,3); 
rho23 = Pest(2,3); 

figure;
subplot(311)
plot(retsTrain(1,:) - mean(retsTrain(1,:)))
subplot(312)
plot(retsTrain(2,:) - mean(retsTrain(2,:)))
subplot(313)
plot(retsTrain(3,:) - mean(retsTrain(3,:)))

figure;
subplot(311)
plot(returns(1,:))
subplot(312)
plot(returns(2,:))
subplot(313)
plot(returns(3,:))

figure;
subplot(311)
plot(sigma1)
subplot(312)
plot(sigma2)
subplot(313)
plot(sigma3)

format long
sigma = diag([sqrt(sigma1(3)), sqrt(sigma2(3)), sqrt(sigma3(3))])*Pest*diag([sqrt(sigma1(3)), sqrt(sigma2(3)), sqrt(sigma3(3))])
format short

%% Estimating portfolio weights
retsVal = Returns(758:end,:)';

N2 = length(Returns(758:end,1));

P = [1, rho12, rho13; rho12, 1, rho23; rho13, rho23, 1];

sigma1Val = zeros(N2 + 1, 1);
sigma2Val = zeros(N2 + 1, 1);
sigma3Val = zeros(N2 + 1, 1);

sigma1Val(1) = sigma1(end);
sigma2Val(1) = sigma2(end);
sigma3Val(1) = sigma3(end);

returnsVal = zeros(3, N2+1);

w = zeros(3,N2);

payoff = zeros(N2,1);

for i = 1:N2
    sigma1Val(i+1) = w1 + a1*(retsVal(1,i)-mu1)^2 + b1*sigma1Val(i);
    sigma2Val(i+1) = w2 + a2*(retsVal(2,i)-mu2)^2 + b2*sigma2Val(i);
    sigma3Val(i+1) = w3 + a3*(retsVal(3,i)-mu3)^2 + b3*sigma3Val(i);
    returnsVal(:,i+1) = diag([sqrt(sigma1Val(i+1)), sqrt(sigma2Val(i+1)), sqrt(sigma3Val(i+1))])*...
        randn(3,1);
    sigma = diag([sqrt(sigma1Val(i+1)), sqrt(sigma2Val(i+1)), sqrt(sigma3Val(i+1))])*...
        P*diag([sqrt(sigma1Val(i+1)), sqrt(sigma2Val(i+1)), sqrt(sigma3Val(i+1))]);

    a = (ones(1,3)/sigma)*returnsVal(:,i+1);
    b = (returnsVal(:,i+1)'/sigma)*returnsVal(:,i+1);
    c = (ones(1,3)/sigma)*ones(3,1);

    u = (gamma - a)/c;

    w(:,i) = ((sigma\returnsVal(:,i+1)) + u*(sigma\ones(3,1)))/gamma;

    %payoff(i) = w(:,i)'*returnsVal(:,i+1);
    %payoff(i) = w(:,i)'*retsVal(:,i);

end
sum(w);
wperc = w*100;

% figure;
% subplot(311)
% plot(retsVal(1,:) - mean(retsVal(1,:)))
% subplot(312)
% plot(retsVal(2,:) - mean(retsVal(2,:)))
% subplot(313)
% plot(retsVal(3,:) - mean(retsVal(3,:)))
% 
% figure;
% subplot(311)
% plot(returnsVal(1,2:end))
% subplot(312)
% plot(returnsVal(2,2:end))
% subplot(313)
% plot(returnsVal(3,2:end))

for j = 1:(N2-1)
    payoff(j) = w(:,j)'*retsVal(:,j+1);
end


% figure;
% plot(payoff)

muStatic = [mean(retsTrain(1,:)); mean(retsTrain(2,:)); mean(retsTrain(3,:))];
cov12Static = cov(retsTrain(1,:), retsTrain(2,:));
cov13Static = cov(retsTrain(1,:), retsTrain(3,:));
cov23Static = cov(retsTrain(2,:), retsTrain(3,:));
varUncondStatic = [var(retsTrain(1,:)), cov12Static(1,2), cov13Static(1,2);...
    cov12Static(1,2), var(retsTrain(2,:)), cov23Static(1,2);...
    cov13Static(1,2), cov23Static(1,2), var(retsTrain(1,:))];


aStatic = (ones(1,3)/varUncondStatic)*muStatic;
bStatic = (muStatic'/varUncondStatic)*muStatic;
cStatic = (ones(1,3)/varUncondStatic)*ones(3,1);

uStatic = (gamma - aStatic)/cStatic;

wStatic = ((varUncondStatic\muStatic) + uStatic*(varUncondStatic\ones(3,1)))/gamma;

payoffStatic = wStatic'*muStatic
mean(payoff)

