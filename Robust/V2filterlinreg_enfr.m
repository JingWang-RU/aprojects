addpath(genpath('./fmethod/'))
% bpen = csvread('./data/IWSLT17.en-fr.tok.tc.bpe.en');
% bpen1 = csvread('./data/IWSLT17.en-fr.tok.tc.bped_1.en');
% bpen2 = csvread('./data/IWSLT17.en-fr.tok.tc.bped_2.en');
% bpen3 = csvread('./data/IWSLT17.en-fr.tok.tc.bped_3.en');
% n = size(bpen, 1);
% m = 1000;
% randex = randperm(n, m);
% 
% X_train = [bpen(randex,:); bpen1(randex,:); bpen2(randex,:); bpen3(randex,:)];
% 
% clear bpen bpen1 bpen2 bpen3;
% bpfr = csvread('./data/IWSLT17.en-fr.tok.tc.bpe.fr');
% bpfr1 = csvread('./data/IWSLT17.en-fr.tok.tc.bped_1.fr');
% bpfr2 = csvread('./data/IWSLT17.en-fr.tok.tc.bped_2.fr');
% bpfr3 = csvread('./data/IWSLT17.en-fr.tok.tc.bped_3.fr');
% y_train = [bpfr(randex,:); bpfr1(randex,:); bpfr2(randex,:); bpfr3(randex,:)];
% N_train = length(y_train);
% clear bpfr bpfr1 bpfr2 bpfr3;

name = "IWSLT17.TED.tst2015_en";
tsten = csvread('./doc2vec/IWSLT17.TED.tst2015.en-fr.tok.tc.bpe.en');
tsten1 = csvread('./doc2vec/IWSLT17.TED.tst2015.en-fr.tok.tc.bped_1.en');
tsten2 = csvread('./doc2vec/IWSLT17.TED.tst2015.en-fr.tok.tc.bped_2.en');
tsten3 = csvread('./doc2vec/IWSLT17.TED.tst2015.en-fr.tok.tc.bped_3.en');
X_test = [tsten; tsten1; tsten2; tsten3];
% y_test = csvread('./doc2vec/IWSLT17.TED.tst2015.en-fr.tok.tc.bpe.fr');
% N_test = length(y_test);
n_test = size(tsten, 1);
% p = 0.1; % 2*p proportion of noisy samples
% [x_robust_mean, y_robust_mean] = robustCentering(X_train, y_train, p);
% X_train = X_train - repmat(x_robust_mean, [N_train, 1]);
% y_train = y_train - y_robust_mean;
% X_test = X_test - repmat(x_robust_mean, [N_test, 1]);
% y_test = y_test - y_robust_mean;

% options = struct();
% options.n_clean = floor(size(X_train,1)/4);
% options.iter = 4;
% options.useG = 1;
% options.filterFnc = 'simple';
% options.debug = 0;
% [xtheta, ~, xerr_test, axc_indices, xac_scores] = filterLinReg(X_train, y_train, X_test, y_test, options);

y_hat = X_test*theta;
y_hat = y_hat - y_robust_mean;

N = size(X_test, 1);
num_iter = 15;
active_indices = (1:N)';
p = 0.2;
for iter = 1:num_iter
    [theta_test, g_test, losses] = v2linReg( X_test(active_indices,:), y_hat(active_indices,:) );
    fprintf("loss %f\n", losses);
    [ind, scores] = filterSimple(g_test, p, mean(g_test));
    active_indices = active_indices(ind);
end
j = 1;
indices = zeros(n_test, 1);
s = zeros(N,1);
s(active_indices) = scores;
for i = 1:4:N
    tmp = s(i: i+3);
    [val, id] = max(tmp);
    indices(j) = id;
%     fprintf("%i %i\n",i,j);
    j = j+1;        
end
fprintf("%d \n", length(active_indices));
a = active_indices;
% histogram(b);
gcf = histogram(indices);
save_file = strcat('v3Filter_', name);

saveas(gcf, strcat('./v3filteredFigures/', save_file, string(m) + '_' + string(p) + '_' + string(num_iter), '.pdf'));
% save(strcat(save_file, '.mat'), 'indices', 'filtered_data', '-v7');
save(strcat('./v3filteredIndex/', save_file, string(m) + '_' + string(p) +'_'+string(num_iter)+'_index.txt'), 'indices', '-ascii');
% csvwrite(strcat(save_file, '_data'), filtered_data);
 
