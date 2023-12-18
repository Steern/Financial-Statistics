function LS = NonLinLSQ(sigma, K, rf, tau, CallS, S)
    N = length(tau);
    
    LS = 0;
    sims = 1000;

    for t = 1:N
        ST = S(t)*exp((rf(t)-sigma(t)*sigma(t)/2)*tau(t)+sigma(t)*sqrt(tau(t))*randn(sims,1));
        CallBS = exp(-rf(t)*tau(t)).*mean(max(ST-K(t,:),0));
        LS = LS + norm(CallS(t,:) - CallBS).^2;
    end
end

