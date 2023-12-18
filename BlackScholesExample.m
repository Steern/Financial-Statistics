% Price Black & Scholes model

%
% This example shows how to price European stock 
% options that expire in three months with an 
% exercise price of $95. Assume that the 
% underlying stock pays no dividend, trades 
% at $100, and has a volatility of 50% per 
% annum. The risk-free rate is 10% per annum.
%
%[Call, Put] = blsprice(100, 95, 0.1, 0.25, 0.5)
%Call = 13.6953
%Put = 6.3497

N=10000000;
S=100;
K=95;
r=0.1;
T=0.25;
sigma=0.5;

% Simulate stock price
ST=S*exp((r-sigma*sigma/2)*T+sigma*sqrt(T)*randn(N,1));
% Compute price
MyCall=exp(-r*T)*mean(max(ST-K,0))