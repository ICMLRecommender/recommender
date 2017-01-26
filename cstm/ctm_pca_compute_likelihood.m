function [log_likelihood, rmse, log_likelihood_rmse, nratings, preds, log_likelihood_priors] = ... 
    ctm_pca_compute_likelihood(doc_or_user, doc, ratings, log_beta_w, sigma_r, omega, omegai, d, theta, var_phi_w, var_zeta, use_zeta_bound, as, sa, ... 
                               lambdaAS, lambdaTheta, lambdaOmegai, lambdaOmega)
    
    %if doc_or_user == 0, keyboard; end
    num_topics = size(as,2);
    N = sum(doc);
    ratings_idx = find(ratings);
    num_ratings = numel(ratings_idx); 

    %g = 0; 
    %g = g - 0.5*lambdaAS * as*as';
    %g = g - N/var_zeta * sum(exp(as)); 
    

    log_likelihood_priors = - 0.5*num_topics*log(2*pi) - 0.5*lambdaAS * as*as';  % P(A|lambdaA)
    if d==1 %doc_or_user == 0
        log_likelihood_priors =  log_likelihood_priors ... 
                           - 0.5*num_topics*log(2*pi) - 0.5*lambdaTheta*theta'*theta ... % P(theta|lambdaTheta)
                           - 0.5*num_topics*log(2*pi) - 0.5*lambdaOmega*omega'*omega; % P(omega|lambdaOmega) 
    end 
    if doc_or_user == 0
        log_likelihood_priors = log_likelihood_priors - 0.5*num_topics*log(2*pi) - 0.5*lambdaOmegai * omegai(d,:)*omegai(d,:)'; % P(omegai|lambdaOmegai) 
    end
    log_likelihood = log_likelihood_priors;

    if use_zeta_bound == 1
        log_likelihood = log_likelihood + N * ( -1/var_zeta * sum(exp(as)) - log(var_zeta)); 
    else
        log_likelihood = log_likelihood + N * ( -log(sum(exp(as)))); 
    end
      
    nratings = numel(ratings_idx); 
    if numel(ratings_idx) ~= 0
        preds=zeros(numel(ratings_idx),1); 
        for i=1:numel(ratings_idx)
            if doc_or_user == 1 
                offset=size(omegai,1)-size(sa,1); 
                preds(i) = (omegai(offset+ratings_idx(i),:) + omega')*as' ...
                        + (sa(ratings_idx(i),:) .* as) * theta; 
            else 
                preds(i) = (omegai(d,:) + omega')*sa(ratings_idx(i),:)' ...
                        + (as .* sa(ratings_idx(i),:)) * theta; 
            end
        end
        se = sum((ratings(ratings_idx) - preds).^2); 
        if doc_or_user == 0, rmse = se; else rmse=0; end
        log_likelihood_rmse = ...  % P(R|U,A,S,theta,omegai,omega)
                         - num_ratings * (log(sigma_r) + 0.5*log(2*pi)) ... 
                         - 1/(2*sigma_r) * ( se );% ...
        log_likelihood = log_likelihood + log_likelihood_rmse; 
        %g = g - 1/(2*sigma_r) * ( se );
        if log_likelihood_rmse>0, fprintf('log_likelihood_rmse is >0: %f\n', log_likelihood_rmse); keyboard; end 

    else
        log_likelihood_rmse=0;
        rmse=0.0; 
        preds=[];
    end

    if numel(var_phi_w) > 0 
        log_likelihood = log_likelihood ... 
             + (as * var_phi_w' )*doc ... 
             + doc' * sum(var_phi_w .* log_beta_w,2) ... % III 
             - doc' * sum(var_phi_w .* log(var_phi_w),2);
    else
        %fprintf('var_phi_w is absent\n'); 
    end
    %g = g + (as * var_phi_w' )*doc; 
    %fprintf('g is %f\n', g); 

    if(isnan(log_likelihood)), fprintf('\nlhood is nan\n'); keyboard; end 
    if(isinf(log_likelihood)), fprintf('\nlhood is inf\n'); keyboard; end 


end
