function [f, df] = fdf_a(a, N, sigma_r, lambdaA, s, ratings, ratings_idx, omega, omegai, theta, var_zeta, var_phi_w_doc, use_zeta_bound)
        f=0; df = zeros(size(a)); 
        f = f - 0.5*lambdaA * a'*a; 
        df = df - lambdaA * a; 

        preds = s(ratings_idx,:)*(omegai' + omega) +  bsxfun(@times, s(ratings_idx,:)', a)'*theta; 
        f = f - 1/(2*sigma_r) * sum((ratings(ratings_idx) - preds).^2); 
        df = df - 1/sigma_r * bsxfun(@times, -s(ratings_idx,:)', theta)*(ratings(ratings_idx)-preds); 

        %preds = zeros(numel(ratings_idx),1); 
        %for i=1:numel(ratings_idx)
        %    pred = (omegai + omega')*s(ratings_idx(i),:)' + theta' * (s(ratings_idx(i),:)' .* a); 
        %    se = se + (ratings(ratings_idx(i)) -pred).^2;
        %    f = f - 1/(2*sigma_r) * (ratings(ratings_idx(i)) -pred).^2; 
        %    df = df - 1/sigma_r * (ratings(ratings_idx(i))-pred)*(-s(ratings_idx(i),:)' .* theta); 
        %    preds(i) = pred; 
        %end


        if use_zeta_bound == 1
            f = f - N/var_zeta * sum(exp(a)); 
            df = df - (N/var_zeta)*exp(a);
        else
            f = f - N*log(sum(exp(a))); 
            df = df - (N/sum(exp(a)))*exp(a);
        end

        f = f + var_phi_w_doc * a;
        df = df + var_phi_w_doc';

        % maximize this function 
        f  = -f;
        df = -df;
end
