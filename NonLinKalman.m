function est = NonLinKalman(y, K, rf, tau, S)
    % Define the state transition function f(x,u) and measurement function h(x)
    %f = @(x, u) [x(1) + u(1); sin(x(2)) + u(2)]; % Example state transition function
    %f = @(x) x;
    %F = @(x) 1;
    F = 1;

    %h = @(x) [x(1)^2; cos(x(2))]; % Example measurement function
    %h = @(x, S, K, rf, tau) S.*normcdf(d1) - K.*exp(-rf*tau).*normcdf(d2);

    % Initialize state estimate and covariance
    %x_hat = [0; 0]; % Initial state estimate
    %P = eye(2); % Initial covariance matrix

    sigPred = 0.1;
    P = 0.1;
    
    % Process and measurement noise covariance matrices
    Q = 0.1; % Process noise covariance
    R = 0.1; % Measurement noise covariance
    
    % Simulation parameters
    N = length(y); % Number of time steps

    % Vectors to store values in
    Xsave = zeros(N ,1); % Stored states
    yhat = zeros(N, 1);
    
    % Simulate the system
    for t = 1:N
%         h = @(sigma) S(t).*normcdf(log(S(t)./K(t,:)) + (rf(t) + sigma/2)*tau(t))...
%             ./ (sqrt(sigma)*sqrt(tau(t))) - K(t,:).*exp(-rf(t)*tau(t))...
%             .*normcdf(log(S(t)./K(t,:)) + (rf(t) + sigma/2)*tau(t))...
%             ./ (sqrt(sigma)*sqrt(tau(t)) - sqrt(sigma)*sqrt(tau(t)));
%         h = @(sigma) S(t).*normcdf(log(S(t)./K(t)) + (rf(t) + sigma/2)*tau(t))...
%             ./ (sqrt(sigma)*sqrt(tau(t))) - K(t).*exp(-rf(t)*tau(t))...
%             .*normcdf(log(S(t)./K(t)) + (rf(t) + sigma/2)*tau(t))...
%             ./ (sqrt(sigma)*sqrt(tau(t)) - sqrt(sigma)*sqrt(tau(t)));
        
        d1 = (log(S(t)/K(t)) + (rf(t) + sigPred^2/2)*tau(t))/(sqrt(sigPred^2)*sqrt(tau(t)));
        d2 = d1 - sqrt(sigPred^2)*sqrt(tau(t));
        yhat(t) = S(t)*normcdf(d1) - K(t)*exp(-rf(t)*tau(t))*normcdf(d2);

        %yhat(t) = h(sigPred) 
        
        Sp = F*P*F' + Q;

%         H = @(sigma) S(t)*sqrt(tau)*normpdf(log(S(t)./K(t,:)) + (rf(t) + sigma/2)*tau(t))...
%             ./ (sqrt(sigma)*sqrt(tau(t)));
%         H = @(sigma) S(t)*sqrt(tau)*normpdf(log(S(t)./K(t)) + (rf(t) + sigma/2)*tau(t))...
%             ./ (sqrt(sigma)*sqrt(tau(t)));
        %Hest = H(sigPred)
        Hest = S(t)*sqrt(tau(t))*normpdf(d1);

        Kt = Sp*Hest'/(Hest*Sp*Hest'+R); % Kalman gain

        sig = sigPred + Kt*(y(t) - yhat(t)); % Updated state estimate
        
        Sp = Sp - Kt*Hest*Sp;

        % Predict the next state
        %sigPred = f(sig);
        sigPred = sig;

        % Store the statevector
        Xsave(t) = sig;
    end

    est = Xsave;
end

