function [log_beta_w, omega, theta] = ... 
    ctm_pca_keywords_m_step(num_docs, doc_or_user, num_ratings, regs, ss_WxT, ss_omega1, ss_omega2, ss_theta1, ss_theta2, log_beta_w, vars, ... 
    lambdaOmega, lambdaTheta, ratings, sigma_r)
% maximize the model parameters 

num_items = nnz(doc_or_user); 
num_users = nnz(doc_or_user==0); 
num_topics = size(ss_omega2,1); 

% update omega
if 1 
%options.method ='lbfgs'; 
%options.display = 0; 
%options.derivativeCheck=1; 
%[omega_grad,f,exitflag,output] = minFunc(@fdf_omega, vars.omega(:), options, sigma_r, lambdaOmega, vars, ratings);

    omega = ss_omega1 / (regs.lambdaOmega*eye(num_topics) + ss_omega2);
    omega = omega'; 

else
    fprintf('not updating omega\n'); 
    omega = vars.omega; 
end


% update theta
if 1 
%options.method ='lbfgs'; 
%options.display = 0; 
%options.derivativeCheck=1; 
%[theta_grad,f,exitflag,output] = minFunc(@fdf_theta, vars.theta(:), options, sigma_r, lambdaTheta, vars, ratings);

    if vars.no_archive == 0
        theta = ss_theta1 / (regs.lambdaTheta*eye(num_topics) + ss_theta2); 
        theta = theta'; 
    else
        theta = vars.theta; 
    end
else 
    fprintf('not updating theta\n'); 
    theta = vars.theta; 
end
%keyboard

% updated topics
update_betas = 1;
if update_betas == 1
    log_beta_w = safelog(ss_WxT) - repmat(safelog(sum(ss_WxT,1)), size(ss_WxT,1), 1);
else 
    fprintf('not updating betas\n'); 
end

end
