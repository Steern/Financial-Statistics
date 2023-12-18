%% Project E2
%% Check data
clc;
clear;
close all;

load("ProjectOptionData.mat");

N = length(OptData);

Sask = zeros(N,1);
Sbid = zeros(N,1);
K = zeros(N,2);
Callask = zeros(N,2);
Callbid = zeros(N,2);
rf = zeros(N,1);
tau = zeros(N,1);

for i = 1:N
    Sask(i,:) = OptData(i).Sask;
    Sbid(i,:) = OptData(i).Sbid;

    K(i,:) = OptData(i).K;
    Callask(i,:) = OptData(i).Callask;
    Callbid(i,:) = OptData(i).Callbid;
%     else
%         K(i,1) = OptData(i).K;
%         K(i,2) = OptData(i).K;
%         Callask(i,1) = OptData(i).Callask;
%         Callask(i,2) = OptData(i).Callask;
%         Callbid(i,1) = OptData(i).Callbid;
%         Callbid(i,2) = OptData(i).Callbid;
%     end
    
    rf(i,:) = OptData(i).rf;
    tau(i,:) = OptData(i).tau;
end

figure(1);
subplot(211)
plot(Sask(:,1));
title("Real S Asks")
subplot(212)
plot(Sbid(:,1));
title("Real S Bids")

figure(2);
plot(K(:,1));
title("Real K's")

figure(3);
subplot(211)
plot(Callask(:,1));
title("Real Call Asks")
subplot(212)
plot(Callbid(:,1));
title("Real Call Bids")

figure(4);
subplot(211)
plot(rf);
title("Short rates")
subplot(212)
plot(tau);
title("Time to maturity")

%% Estimation (Constant sigma)
sigmaInit = 0.1;

K1 = K;
CallS = Callask;
S = Sask;

LS = @(params) NonLinLSQConst2(params, K1, rf, tau, CallS, S);

sigmaEst = fminsearch(LS, sigmaInit);

disp('Estimated Sigma using fminsearch:');
disp(sigmaEst);

CallBS = zeros(N,2);

% for t = 1:N
%     ST = S(t)*exp((rf(t)-sigmaEst*sigmaEst/2)*tau(t)+sigmaEst*sqrt(tau(t))*randn(N,1));
%     CallBS(t,:) = exp(-rf(t)*tau(t)).*mean(max(ST-K1(t,:),0));
% end

for t = 1:N
    d1 = (log(S(t)./K(t,:)) + (rf(t) + sigmaEst/2)*tau(t)) ./ (sqrt(sigmaEst)*sqrt(tau(t)));
    d2 = d1 - sqrt(sigmaEst)*sqrt(tau(t));
    CallBS(t,:) = S(t).*normcdf(d1) - K(t,:).*exp(-rf(t)*tau(t)).*normcdf(d2);
end

figure;
subplot(211)
plot([CallS(:,1), CallBS(:,1)])
title("Real Call Price vs Estimated Call Price for option 1")
subplot(212)
resid1 = CallS(:,1) - CallBS(:,1);
plot(resid1)
title("Residuals")

figure;
subplot(211)
plot([CallS(:,2), CallBS(:,2)])
title("Real Call Price vs Estimated Call Price for option 2")
subplot(212)
resid2 = CallS(:,2) - CallBS(:,2);
plot(resid2)
title("Residuals")

figure;
subplot(211)
normplot(resid1)
title("Normplot of Residual for option 1")
subplot(212)
normplot(resid2)
title("Normplot of Residual for option 2")


%% Estimation (Time-varying sigma)
sigmaInit = 0.1*ones(N,1);

K1 = K;
CallS = Callask;
S = Sask;
%sigma = sigmaInit;
% for t = 1:1
%     d1 = (log(S(t)./K(t,:)) + (rf(t) + sigma(t)/2)*tau(t)) ./ (sqrt(sigma(t))*sqrt(tau(t)));
%     d2 = d1 - sqrt(sigma(t))*sqrt(tau(t));
%     CallBS = S(t).*normcdf(d1) - K(t,:).*exp(-rf(t)*tau(t)).*normcdf(d2);
%     LS = LS + norm(CallS(t,:) - CallBS).^2;
% end

LS = @(params) NonLinLSQ2(params, K1, rf, tau, CallS, S);

%options = optimset("Display", "iter", "MaxIter", 1000);
options = optimset("Display", "iter");
LB = zeros(N,1);
sigmaEst = fminsearchbnd(LS, sigmaInit, LB, [], options);

%disp('Estimated Sigma using fminsearch:');
%disp(sigmaEst);

CallBS = zeros(N,2);

% for t = 1:N
%     ST = S(t)*exp((rf(t)-sigmaEst(t)*sigmaEst(t)/2)*tau(t)+sigmaEst(t)*sqrt(tau(t))*randn(N,1));
%     CallBS(t,:) = exp(-rf(t)*tau(t)).*mean(max(ST-K1(t,:),0));
% end
for t = 1:N
    d1 = (log(S(t)./K(t,:)) + (rf(t) + sigmaEst(t)/2)*tau(t)) ./ (sqrt(sigmaEst(t))*sqrt(tau(t)));
    d2 = d1 - sqrt(sigmaEst(t))*sqrt(tau(t));
    CallBS(t,:) = S(t).*normcdf(d1) - K(t,:).*exp(-rf(t)*tau(t)).*normcdf(d2);
end

figure;
subplot(211)
plot([CallS(:,1), CallBS(:,1)])
title("Real Call Price vs Estimated Call Price for option 1")
subplot(212)
resid1 = CallS(:,1) - CallBS(:,1);
plot(resid1)
title("Residuals")

% Find indices for Option 2
idx2 = find(K(:,1) ~= K(:,2));
CallS2 = CallS(idx2,2);
CallBS2 = CallBS(idx2,2);

figure;
subplot(211)
plot([CallS2, CallBS2])
title("Real Call Price vs Estimated Call Price for option 2")
subplot(212)
resid2 = CallS2 - CallBS2;
plot(resid2)
title("Residuals")


% figure;
% subplot(211)
% plot([CallS(:,2), CallBS(:,2)])
% title("Real Call Price vs Estimated Call Price for option 2")
% subplot(212)
% resid2 = CallS(:,2) - CallBS(:,2);
% plot(resid2)
% title("Residuals")

figure;
subplot(211)
normplot(resid1)
title("Normplot of Residual for option 1")
subplot(212)
normplot(resid2)
title("Normplot of Residual for option 2")

figure;
plot(sigmaEst)
title("Estimate of the volatilty (sigma squared)")

%% Non Linear Kalman (1 dim)
K1 = K(:,1);
CallS = Callask(:,1);
S = Sask;

sigmaEstK = NonLinKalman(CallS, K1, rf, tau, S);
sigmaEstK = sigmaEstK.^2;

CallBS = zeros(N,1);

for t = 1:N
    d1 = (log(S(t)./K(t)) + (rf(t) + sigmaEstK(t)/2)*tau(t)) ./ (sqrt(sigmaEstK(t))*sqrt(tau(t)));
    d2 = d1 - sqrt(sigmaEstK(t))*sqrt(tau(t));
    CallBS(t) = S(t).*normcdf(d1) - K(t).*exp(-rf(t)*tau(t)).*normcdf(d2);
end

figure;
subplot(211)
plot([CallS, CallBS])
title("Real Call Price vs Estimated Call Price for option 1")
subplot(212)
resid1 = CallS - CallBS;
plot(resid1)
title("Residuals")

% figure;
% subplot(211)
% plot([CallS(:,2), CallBS(:,2)])
% title("Real Call Price vs Estimated Call Price for option 2")
% subplot(212)
% resid2 = CallS(:,2) - CallBS(:,2);
% plot(resid2)
% title("Residuals")

figure;
%subplot(211)
normplot(resid1)
title("Normplot of Residual for option 1")
% subplot(212)
% normplot(resid2)
% title("Normplot of Residual for option 2")

figure;
plot(sigmaEstK)
title("Estimate of the volatilty (sigma squared)")

%% Non Linear Kalman (Full)
K1 = K;
CallS = Callask;
S = Sask;

sigmaEstK = NonLinKalmanFull(CallS, K1, rf, tau, S);
sigmaEstK = sigmaEstK.^2;

CallBS = zeros(N,2);

for t = 1:N
    d1 = (log(S(t)./K(t,:)) + (rf(t) + sigmaEstK(t)/2)*tau(t)) ./ (sqrt(sigmaEstK(t))*sqrt(tau(t)));
    d2 = d1 - sqrt(sigmaEstK(t))*sqrt(tau(t));
    CallBS(t,:) = S(t).*normcdf(d1) - K(t,:).*exp(-rf(t)*tau(t)).*normcdf(d2);
end

figure;
subplot(211)
plot([CallS(:,1), CallBS(:,1)])
title("Real Call Price vs Estimated Call Price for option 1")
subplot(212)
resid1 = CallS(:,1) - CallBS(:,1);
plot(resid1)
title("Residuals")

% Find indices for Option 2
idx2 = find(K(:,1) ~= K(:,2));
CallS2 = CallS(idx2,2);
CallBS2 = CallBS(idx2,2);

figure;
subplot(211)
plot([CallS2, CallBS2])
title("Real Call Price vs Estimated Call Price for option 2")
subplot(212)
resid2 = CallS2 - CallBS2;
plot(resid2)
title("Residuals")

figure;
subplot(211)
normplot(resid1)
title("Normplot of Residual for option 1")
subplot(212)
normplot(resid2)
title("Normplot of Residual for option 2")

figure;
plot(sigmaEstK)
title("Estimate of the volatilty (sigma squared)")

%% Non Linear Kalman (Full Iterated)
K1 = K;
CallS = Callask;
S = Sask;

sigmaEstK = NonLinIterKalmanFull(CallS, K1, rf, tau, S);
sigmaEstK = sigmaEstK.^2;

CallBS = zeros(N,2);

for t = 1:N
    d1 = (log(S(t)./K(t,:)) + (rf(t) + sigmaEstK(t)/2)*tau(t)) ./ (sqrt(sigmaEstK(t))*sqrt(tau(t)));
    d2 = d1 - sqrt(sigmaEstK(t))*sqrt(tau(t));
    CallBS(t,:) = S(t).*normcdf(d1) - K(t,:).*exp(-rf(t)*tau(t)).*normcdf(d2);
end

figure;
subplot(211)
plot([CallS(:,1), CallBS(:,1)])
title("Real Call Price vs Estimated Call Price for option 1")
subplot(212)
resid1 = CallS(:,1) - CallBS(:,1);
plot(resid1)
title("Residuals")

% figure;
% subplot(211)
% plot([CallS(:,2), CallBS(:,2)])
% title("Real Call Price vs Estimated Call Price for option 2")
% subplot(212)
% resid2 = CallS(:,2) - CallBS(:,2);
% plot(resid2)
% title("Residuals")

figure;
%subplot(211)
normplot(resid1)
title("Normplot of Residual for option 1")
% subplot(212)
% normplot(resid2)
% title("Normplot of Residual for option 2")

figure;
plot(sigmaEstK)
title("Estimate of the volatilty (sigma squared)")

