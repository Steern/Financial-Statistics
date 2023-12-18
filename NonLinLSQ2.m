function LS = NonLinLSQ2(sigma, K, rf, tau, CallS, S)
    N = length(tau);
    
    LS = 0;

    for t = 1:N
        if sigma(t) <= 0
            sigma(t) = 1e-5;
        end
        if K(t,1) == K(t,2)
            d1 = (log(S(t)./K(t,1)) + (rf(t) + sigma(t)/2)*tau(t)) ./ (sqrt(sigma(t))*sqrt(tau(t)));
            d2 = d1 - sqrt(sigma(t))*sqrt(tau(t));
            CallBS = S(t).*normcdf(d1) - K(t,1).*exp(-rf(t)*tau(t)).*normcdf(d2);
            LS = LS + norm(CallS(t,1) - CallBS).^2;
        else 
            d1 = (log(S(t)./K(t,:)) + (rf(t) + sigma(t)/2)*tau(t)) ./ (sqrt(sigma(t))*sqrt(tau(t)));
            d2 = d1 - sqrt(sigma(t))*sqrt(tau(t));
            CallBS = S(t).*normcdf(d1) - K(t,:).*exp(-rf(t)*tau(t)).*normcdf(d2);
            LS = LS + norm(CallS(t,:) - CallBS).^2;
        end
    end
end

