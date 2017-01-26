function [log_likelihood_all, log_likelihood_rmse_all, ss_WxT, ss_omega1, ss_omega2, ss_theta1, ss_theta2, vars] = ... 
  ctm_pca_keywords_e_step(WxD, doc_or_user, ratings, num_topics, log_beta_w, sigma_r, vars, regs, max_iter_e_step)

    update_topics=1; 
    
    if ~isfield(vars,'new_user'), vars.new_user = 0; end
    if ~isfield(vars,'new_item'), vars.new_item = 0; end
    if ~isfield(vars,'no_archive'), vars.no_archive = 0; end
    if vars.no_archive == 1, fprintf('not using archive\n'); end 
    
    [num_terms, num_docs] = size(WxD); 
    max_iter_e_step=1; 
    if ~exist('max_iter_e_step','var'), max_iter_e_step = 1; end
    for iter_e_step = 1:max_iter_e_step

    if iter_e_step == 1
        if ~isfield(vars,'do_no_init') || vars.do_no_init == 0
            fprintf('initializing some variational variables\n'); 
            vars.as = ones(num_docs, num_topics)*1e-2; 
            vars.omegai = ones(num_docs, num_topics)*1e-2; 
            if ~isfield(vars,'omega'), vars.omega = ones(num_topics,1)*1e-2; end
            if ~isfield(vars,'theta'), vars.theta = ones(num_topics,1)*1e-2; end
        else 
            fprintf('not initializing the model variables\n'); 
            %update_topics = 0; 
        end
        vars.var_zeta   = 10*ones(num_docs,1); 
    end
    if vars.no_archive == 1 
        % set users' archives to 0
        vars.as(find(doc_or_user==0),:) = 0;
    end

    as = vars.as;
    omegai = vars.omegai;
    omega = vars.omega; 
    theta = vars.theta; 
    var_zeta   = vars.var_zeta;   
    lambdaA = regs.lambdaA; 
    lambdaS = regs.lambdaS; 
    lambdaTheta = regs.lambdaTheta; 
    lambdaOmega = regs.lambdaOmega; 
    lambdaOmegai = regs.lambdaOmegai; 
    use_zeta_bound = vars.use_zeta_bound;

    ss_WxT = zeros(size(WxD,1), num_topics); 
    ss_omega1 = 0; 
    ss_omega2 = 0; 
    ss_theta1 = 0; 
    ss_theta2 = 0; 

    if update_topics == 0, fprintf('not updating topics\n'); else fprintf('updating topics\n'); end 

    log_likelihood_all = 0; rmse_tot=0; log_likelihood_rmse_all=0;
    for d=1:num_docs

       if doc_or_user(d) == 1
            sa = as(doc_or_user==0, :); 
            didx = nnz(d>=find(doc_or_user==1)); 
            ratings_d = ratings(:,didx);
            lambdaAS = lambdaS; 
       else
            %if doc_or_user(d-1) == 1, fprintf('start of users\n'); keyboard; end
            sa = as(doc_or_user==1, :); 
            didx = nnz(d>=find(doc_or_user==0)); 
            ratings_d = ratings(didx,:)';
            lambdaAS = lambdaA; 
       end

       if vars.new_user == 1 && doc_or_user(d) == 1 && vars.new_item == 1
          continue;
       end


       ratings_idx = find(ratings_d);
       if nnz(ratings_d)==0, fprintf('no ratings\n'); end

       fprintf('doc %d\r', d); 
       %% Init the variational variables.
       if vars.new_user == 1 && isfield(vars, 'var_phi_w')
           var_phi_w = vars.var_phi_w;
       else
           var_phi_w = ones(num_terms, num_topics)/num_topics; % variational var for z^w
                                                  % !!! why not divide by num_terms? 
                                                  %     the above init. corresponds to what's done in D.Blei's LDA-C code
       end

        if 0 
           [ll, rmse, log_likelihood_rmse] = ... 
                    ctm_pca_compute_likelihood(doc_or_user(d), WxD(:,d), ratings_d, log_beta_w, sigma_r, omega, omegai, d, theta, var_phi_w, ... 
                                    var_zeta(d), use_zeta_bound, ... 
                                    as(d,:), sa, lambdaAS, lambdaTheta, lambdaOmegai, lambdaOmega);
           fprintf('%d: %f\n', d, ll); 
           fprintf('\n%d: log_likelihood_rmse before: %f\n', d, log_likelihood_rmse); 
        end 

       log_likelihood_old = eps; 
       converged=1.; iter=0; iters_newton_tot=0; iters_cg_tot=0;
       words_per_doc = sum(WxD,1); 
       time_doc = tic; 
       while converged>1e-5

          iter=iter+1;

          tid = WxD(:,d)>0; 

          iters_cg=0;iters_newton=0;
          % optimize variational parameters
          var_opt_str='initialize'; check_opt; 
          if use_zeta_bound == 1 && (vars.new_user == 0 || doc_or_user(d)==0)   ... 
                                && (vars.no_archive == 0 || doc_or_user(d) == 1)
              var_zeta(d) = optimize_zeta(as(d,:)); 
              var_opt_str='var_zeta'; check_opt; 
          end

          if doc_or_user(d) == 1 
              if vars.new_user==0
              if 1 %update_topics == 1
                  as(d,:) = optimize_s(as(d,:), ratings_d, WxD(:,d), sa, omega, omegai, theta, sigma_r, lambdaS, var_phi_w, var_zeta(d), use_zeta_bound);
                  var_opt_str='s'; check_opt; 
              end
              end
          else
              if vars.no_archive == 0
                  if 1 %update_topics == 1
                      b = as(d,:); 
                      % NEW
                      % once there are ratings, remove topics 
                      %keyboard
                      %if nnz(ratings_d) ~= 0 
                      %var_phi_w = zeros(size(var_phi_w)); 
                      %var_zeta(d) = 1; 
                      %end 
                      % END OF NEW
                      as(d,:) = optimize_a(as(d,:), ratings_d, WxD(:,d), sa, omega, omegai(d,:), theta, sigma_r, lambdaA, var_phi_w, var_zeta(d), use_zeta_bound); 
                      var_opt_str='a'; check_opt; 
                  end
              else
                as(d,:) = zeros(size(as(d,:)));
              end
              c = omegai(d,:); 
              omegai(d,:) = optimize_omegai(as(d,:), ratings_d, sa, omega, omegai(d,:), theta, sigma_r, lambdaOmegai); 
              var_opt_str='omegai'; check_opt; 
          end
          if use_zeta_bound == 1 && (vars.new_user==0 || doc_or_user(d) == 0) ... 
                                && (vars.no_archive == 0 || doc_or_user(d) == 1)
              var_zeta(d) = optimize_zeta(as(d,:)); 
              var_opt_str='var_zeta'; check_opt; 
          end

          if update_topics == 1 && (vars.new_user == 0 || doc_or_user(d)==0) ... 
                                && (vars.no_archive == 0 || doc_or_user(d) == 1)
              if nnz(WxD(:,d)) ~= 0 
                  var_phi_w = optimize_phi(WxD(:,d), log_beta_w, as(d,:)); 
                  var_opt_str='var_phi_w'; check_opt; 
              end
          end 
                   
          [log_likelihood, rmse, log_likelihood_rmse] = ... 
                ctm_pca_compute_likelihood(doc_or_user(d), WxD(:,d), ratings_d, log_beta_w, sigma_r, omega, omegai, d, theta, var_phi_w, ... 
                               var_zeta(d), use_zeta_bound, as(d,:), sa, ... 
                               lambdaAS, lambdaTheta, lambdaOmegai, lambdaOmega);
          %if d == 862, keyboard; end

          if iter == 1
              fprintf('\rdoc %d (%d words), log likelihood %.6f, rmse: %.3f', d, full(sum(WxD(:,d))), log_likelihood, sqrt(rmse/nnz(ratings_d)));
          end

          % check convergence
          converged = (log_likelihood_old-log_likelihood)/log_likelihood_old; 
          log_likelihood_old = log_likelihood; 

          if iter==1, rmse_one = rmse; end
          iters_newton_tot=iters_newton_tot+iters_newton; iters_cg_tot=iters_cg_tot+iters_cg;
       end % while not converged
       fprintf('\rdoc %d (%d words), log likelihood %.6f, rmse: %.3f', d, full(sum(WxD(:,d))), log_likelihood, sqrt(rmse/nnz(ratings_d)));

       %fprintf('\niter: %d, converged %f\n', iter, converged); 
       %pause

       if rmse_one>rmse, rmse_str='down';, elseif rmse_one<rmse, rmse_str='up'; else, rmse_str='neutral'; end
       fprintf(' %s (%.3f),  (%d iters, iters_cg %.1f, iters_newton %.1f, %.2fs)\n', rmse_str, rmse-rmse_one, iter, iters_cg_tot/iter, iters_newton_tot/iter, toc(time_doc)); 

       if doc_or_user(d)==1, log_likelihood=log_likelihood-log_likelihood_rmse; end % don't count ratings' likelihood twice
       log_likelihood_all = log_likelihood_all + log_likelihood;
       if doc_or_user(d)==0, log_likelihood_rmse_all = log_likelihood_rmse_all + log_likelihood_rmse; end
       %fprintf('\n%d: log_likelihood_rmse after: %f\n', d, log_likelihood_rmse); 

       % save some sufficient statistics needed by the m-step 
       if doc_or_user(d) == 0, 
           for i=1:numel(ratings_idx)
                ss_omega1 = ss_omega1 + ... 
                (ratings_d(ratings_idx(i)) - (sa(ratings_idx(i),:).*as(d,:))*theta - omegai(d,:)*sa(ratings_idx(i),:)')*(sa(ratings_idx(i),:));
            end
            if numel(ratings_idx) > 0 
                ss_omega2 = ss_omega2 + sa(ratings_idx,:)'*sa(ratings_idx,:);
            end

           for i=1:numel(ratings_idx)
                ss_theta1 = ss_theta1 + ... 
                (ratings_d(ratings_idx(i)) - (omegai(d,:)+omega')*sa(ratings_idx(i),:)')*(sa(ratings_idx(i),:).*as(d,:));

                ss_theta2 = ss_theta2 + (sa(ratings_idx(i),:).*as(d,:))' * (sa(ratings_idx(i),:).*as(d,:));
            end
       end

       if doc_or_user(d) == 1 || vars.no_archive == 0 
           ss_WxT       = ss_WxT       + repmat(WxD(:,d), 1, num_topics).*var_phi_w;
       end

       if doc_or_user(d) == 0
          rmse_tot = rmse_tot+rmse; 
       end
       if vars.new_user == 1 
           %vars.var_phi_w = var_phi_w;
       end

    end % for each document

    ss_omega1 = ss_omega1 * (1/sigma_r); 
    ss_omega2 = ss_omega2 * (1/sigma_r); 
    ss_theta1 = ss_theta1 * (1/sigma_r); 
    ss_theta2 = ss_theta2 * (1/sigma_r); 


    vars.var_zeta  = var_zeta;
    vars.as = as;
    vars.omegai = omegai;
    vars.omega = omega; 
    vars.theta = theta; 
    %vars.var_phi_w = var_phi_w; 

    fprintf('\nmse_avg: %f\n', rmse_tot/nnz(ratings)); 
    fprintf('end of iteration %d/%d e step\n', iter_e_step, max_iter_e_step); 
    end
end

function [omegai] = optimize_omegai(as, ratings, sa, omega, omegai, theta, sigma_r, lambdaOmegai) 

    num_topics = numel(omega); 

    ratings_idx = find(ratings); 

    %term1 = 0;
    %for i=1:numel(ratings_idx)
    %    term1 = term1 + ... 
    %        (ratings(ratings_idx(i)) - (sa(ratings_idx(i),:).*as)*theta - (omega')*sa(ratings_idx(i),:)')*sa(ratings_idx(i),:);
    %end
    term1 = (ratings(ratings_idx) - bsxfun(@times, sa(ratings_idx,:), as)* theta - sa(ratings_idx,:)*omega)' * sa(ratings_idx,:); 

    warning('error', 'MATLAB:nearlySingularMatrix');
    try
        omegai = (-1/sigma_r * term1) / (- sa(ratings_idx,:)'*sa(ratings_idx,:)/sigma_r - lambdaOmegai*eye(num_topics)); 
    catch
       fprintf('caught warning\n'); 
       keyboard
    end
 
    omegai=omegai';

end

function [a] = optimize_a(a, ratings, doc, s, omega, omegai, theta, sigma_r, lambdaA, var_phi_w, var_zeta, use_zeta_bound) 

    var_phi_w_doc = doc'*var_phi_w; 
    N = sum(doc); 

    options.method ='lbfgs'; 
    options.cgUpdate=1; 
    %options.optTol = 1e-3; 
    %options.progTol = 1e-3; 
    options.display = 0; 
    options.MaxIter = 5000;
    options.MaxFunEvals = 5000; 
    %options.derivativeCheck=1; 

    [a,f,exitflag,output] = minFunc(@fdf_a, a(:), options, N, sigma_r, lambdaA, s, ratings, find(ratings), omega, omegai, theta, var_zeta, var_phi_w_doc, use_zeta_bound);
    %fprintf('minFunc (%d iters) %f\n', output.iterations, toc); 
    if exitflag<0, fprintf(output.message); keyboard; end
    if exitflag==0, fprintf(output.message); keyboard; end
    iters_tot = output.iterations; 
    a=a'; 

end

function [s] = optimize_s(s, ratings, doc, a, omega, omegai, theta, sigma_r, lambdaS, var_phi_w, var_zeta, use_zeta_bound) 

    var_phi_w_doc = doc'*var_phi_w; 
    N = sum(doc); 

    options.method ='lbfgs'; 
    options.cgUpdate=1; 
    %options.optTol = 1e-3; 
    %options.progTol = 1e-3; 
    options.display = 0; 
    options.MaxIter = 5000;
    options.MaxFunEvals = 5000; 

    %options.derivativeCheck=1; 

    [s,f,exitflag,output] = minFunc(@fdf_s, s(:), options, N, sigma_r, lambdaS, a, ratings, find(ratings), omega, omegai, theta, var_zeta, var_phi_w_doc, use_zeta_bound);
    %fprintf('minFunc (%d iters) %f\n', output.iterations, toc); 
    if exitflag<0, fprintf(output.message); keyboard; end
    if exitflag==0, fprintf(output.message); keyboard; end
    iters_tot = output.iterations; 
    s=s'; 

end

function [var_zeta] = optimize_zeta(as)
    var_zeta = sum(exp(as)); 
end

function var_phi_tot = optimize_phi(doc, log_beta, var_omega)
    % Optimize either of the var_phi_{s,w}

    n=find(doc);
    %var_phi = zeros(numel(n), size(log_beta,2)); 

    %keyboard
    var_phi = bsxfun(@plus, log_beta(n,:), var_omega); 
    phisum = log_sum_mat(var_phi);
    %var_phi(n,:) = exp(var_phi(n,:) - phisum); % var_phi in log-space
    var_phi = exp( bsxfun(@minus, var_phi, phisum)); % var_phi in log-space
    if any(sum(var_phi,2) > (1+1e-3)) || any(sum(var_phi,2) < (1-1e-3)), fprintf('var_phi not sum to 1\n'); keyboard; end 

    var_phi_tot = zeros(size(log_beta)); 
    var_phi_tot(n,:) = var_phi; 

%    for n=find(doc)'
%        % lda-inference.c:56
%        var_phi(n,:) = var_omega + log_beta(n,:); 
%        %phisum = var_phi(n,1);% phi already in log-space 
%        %for k=2:num_topics
%        %   phisum = log_sum(phisum, var_phi(n,k));
%        %end
%        phisum = log_sum_mat(var_phi(n,:));
%        % normalize var_phi_w              
%        var_phi(n,:) = exp(var_phi(n,:) - phisum); % var_phi in log-space
%        if sum(var_phi(n,:)) > (1+1e-3) || sum(var_phi(n,:)) < (1-1e-3), fprintf('var_phi not sum to 1\n'); keyboard; end 
%    end

    if all(abs(var_phi(:)) < 1e-4), fprintf('var_phi is small\n'); keyboard; end


    %var_phi = exp(uv).*beta_w; 
%    log_var_phi_w = repmat(uv, size(beta_w,1), 1) + log(beta_w); 
%    phisum = log_var_phi_w(:,1);
%    for k=2:num_topics
%       log_phisum = log_sum(phisum, log_var_phi_w(:,k));
%    end
%
%    var_phi_w = exp(bsxfun(@minus, log_var_phi_w, log_phisum));

end

