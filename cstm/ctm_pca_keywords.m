% Implementation of the CSTM model
%
% NOTE: This file only contains the inference/learning code. I.e., it assumes
% that the data is loaded and several hyper-parameters are preset.  
% 
% In this particular implementation everything lives in the K-1 dimensions. 
addpath('~/bin/minFunc_2012/minFunc/'); % for the optimizer
addpath('~/bin/minFunc_2012/autoDif/'); 
addpath('~/bin/minFunc_2012/minFunc/compiled/'); 
addpath('~/bin/lightspeed/');
addpath('../utils');

data_params.use_toy_dataset = 0;

use_keyword_weighting = 0; 
[WxD, KxD, WxR, KxR, ratings, words, keywords] = get_data(data_params.dataset, use_keyword_weighting);
use_toy=0; 
if use_toy == 1
    num_docs=60; num_users=50; 
    ratings = ratings(1:num_users,1:num_docs); 
    WxD = WxD(:,1:num_docs); 
    WxR = WxR(:,1:num_users); 
    KxD = KxD(:,1:num_docs); 
    KxR = KxR(:,1:num_users); 
end


[data_params, WxDtrain, WxDtest, KxDtrain, KxDtest, WxRtrain, WxRtest, KxRtrain, KxRtest, doc_or_user, rTrain, rValid, rTest, words, keywords] ... 
            = prep_data(data_params, WxD, KxD, WxR, KxR, ratings, words, keywords);
num_topics = data_params.num_topics;

if numel(rValid) == 0 
   rValid = rTest;  
end 

WxD=WxDtrain; 
KxD=KxDtrain;
%WxR=WxRtrain; 
%KxR=WxRtrain; 
ratings=rTrain;

if isfield(data_params, 'use_words') && data_params.use_words == 0 
    fprintf('no using any words\n'); 
    WxD = []; 
end

%% deal with users just as if they were any other document (except they have no
%% word)
%if numel(KxRtrain) > 0
%    WxDtrain = WxDtrain(:,1:10); 
%    KxDtrain = [KxDtrain  KxRtrain]; 
%    KxDtest = [KxDtest  KxRtest]; 
%end


% model parameters: 
%   beta_w (num_terms, num_topics) 
%   lambdaS, lambdaA, lambdaTheta, lambdaOmega, lambdaOmegai (scalars)
% variational parameters: 
%   var_zeta    (num_docs,1)
%   var_phi_w   (num_terms, num_topics)
%   theta  (num_topics,1)
%   omega  (num_topics,1)
%   omegai (num_topics, num_users)
%   as   (num_topics, num_docs)

if ~isfield(data_params,'use_keywords')
    use_keywords = 1; 
else
    use_keywords = data_params.use_keywords; 
end
num_terms = size(WxD,1); 
num_docs     = size(WxD,2); 
max_length_corpus = max(sum(WxD>0, 1));

%% TODO: check how to properly initialize these 
% check ctm.c:random_init()
% Model parameters
beta_w = rand(num_terms, num_topics)*1e-2; 
beta_w = beta_w./repmat(sum(beta_w,1), size(beta_w,1), 1); 
log_beta_w = log(beta_w); 
% covariance matrices below are stored as full but updated as diagonal
fprintf('initializing hyper-parameters (inv_sigma_u, inv_sigma_v, sigma_r)\n'); 
regs.lambdaA = 0.02; 
regs.lambdaS = 0.01; 
regs.lambdaTheta = 0.03; 
regs.lambdaOmega = 0.04; 
regs.lambdaOmegai = 0.05; 
sigma_r = 1; 

debugging=1;

log_likelihood_old=-inf; 
converged = inf; 
iter_em = 1; 

vars.var_as  = zeros(num_docs, num_topics); 
vars.omega = zeros(num_topics, 1); 
vars.omegai = zeros(num_docs, num_topics); 
vars.theta = zeros(num_topics, 1); 
vars.var_zeta   = 10*ones(num_docs,1); 

if ~exist('max_iter_e_step','var'), max_iter_e_step = 1; end

valid_loss_min=Inf; 
while (converged > 1e-4 || converged < 0) && iter_em < 500

    fprintf('iteration: %d\n', iter_em); 
    %if iter_em>1 && iter_em<5, help_check_entropy; end 

    fprintf('e-step\n'); 
    [log_likelihood, ss_WxT, ss_omega1, ss_omega2, ss_theta1, ss_theta2, vars] = ... 
          ctm_pca_keywords_e_step(WxD, doc_or_user, ratings, num_topics, log_beta_w, sigma_r, vars, regs, max_iter_e_step);

    fprintf('m-step\n'); 
    [log_beta_w, omega, theta] = ... 
       ctm_pca_keywords_m_step(num_docs, doc_or_user, nnz(ratings), regs, ss_WxT, ss_omega1, ss_omega2, ss_theta1, ss_theta2, ... 
                               log_beta_w); 

    converged = abs(log_likelihood_old-log_likelihood)/abs(log_likelihood_old); 
    if isnan(converged), converged = inf; end 
    fprintf('\t%d: ll %.3f, conv. %.6f\n', iter_em, log_likelihood, converged)
    if log_likelihood_old>log_likelihood, fprintf('ll decreases\n'); keyboard; end 
    log_likelihood_old = log_likelihood; 

    iter_em = iter_em + 1; 

    if debugging == 1, print_top_topical_terms(words, [], 0, num_topics, log_beta_w, []); end

    % check test error 
    preds_test = 0; 
    preds_train = get_preds(ratings, vars, doc_or_user); 
    preds_valid = get_preds(rValid, vars, doc_or_user); 
    preds_test = get_preds(rTest, vars, doc_or_user); 

    idx = find(ratings); 
    train_loss = sum((ratings(idx)-preds_train).^2)/numel(idx); 
    fprintf('train loss %f\n', train_loss); 
    idx = find(rValid); 
    valid_loss = sum((rValid(idx)-preds_valid).^2)/numel(idx); 
    fprintf('Valid loss %f\n', valid_loss); 
    idx = find(rTest); 
    test_loss = sum((rTest(idx)-preds_test).^2)/numel(idx); 
    fprintf('test loss %f\n', test_loss); 
    if valid_loss < valid_loss_min
        valid_loss_min = valid_loss; 
        fprintf('min loss attained\n'); 
    end

    if valid_loss > 50 
        break 
    end
end

fprintf('Final: iters (%d), ll %.3f\n', iter_em, log_likelihood)
