function LS = NonLinLSQConst2(sigma, K, rf, tau, CallS, S)
    N = length(tau);
    
    LS = 0;

    for t = 1:N
        if sigma <= 0
            sigma = 1e-5;
        end
        d1 = (log(S(t)./K(t,:)) + (rf(t) + sigma/2)*tau(t)) ./ (sqrt(sigma)*sqrt(tau(t)));
        d2 = d1 - sqrt(sigma)*sqrt(tau(t));
        CallBS = S(t).*normcdf(d1) - K(t,:).*exp(-rf(t)*tau(t)).*normcdf(d2);
        LS = LS + norm(CallS(t,:) - CallBS).^2;
    end
end

