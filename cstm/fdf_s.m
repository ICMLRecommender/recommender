function [f, df] = fdf_s(s, N, sigma_r, lambdaS, a, ratings, ratings_idx, omega, omegai, theta, var_zeta, var_phi_w_doc, use_zeta_bound)

        f=0; df = zeros(size(s)); 

        f = f - 0.5*log(lambdaS) - 0.5*lambdaS * s'*s; 
        df = df - lambdaS * s; 

        offset=size(omegai,1)-size(a,1); 

        preds = bsxfun(@plus, omegai(offset+ratings_idx,:), omega')*s + bsxfun(@times, a(ratings_idx,:)', s)'*theta; 
        f = f - 1/(2*sigma_r) * sum((ratings(ratings_idx) - preds).^2); 
        df = df - 1/sigma_r * (bsxfun(@minus, bsxfun(@times, -a(ratings_idx,:)', theta), omega) - omegai(offset+ratings_idx,:)')*(ratings(ratings_idx)-preds);
        %for i=1:numel(ratings_idx)
        %    pred = (omegai(ratings_idx(i),:) + omega')*s + theta' * (a(ratings_idx(i),:)' .* s); 
        %    f = f - 1/sigma_r * (ratings(ratings_idx(i)) - pred).^2; 
        %    df = df - 1/sigma_r * 2*(ratings(ratings_idx(i))-pred)*(-a(ratings_idx(i),:)' .* theta  - omega - omegai(ratings_idx(i),:)'); 
        %end
        %keyboard

        if use_zeta_bound==1 
            f = f - N/var_zeta * sum(exp(s)); 
            df = df - N/var_zeta*exp(s);
        else
            f = f - N*log(sum(exp(s))); 
            df = df - (N/sum(exp(s)))*exp(s);
        end

        f = f + var_phi_w_doc * s;
        df = df + var_phi_w_doc';

        % maximize this function 
        f  = -f;
        df = -df;
end
