
% load data
name = 'IWSLT17.TED.tst2015_fr';

save_file = strcat('./v2filtered/v2Filter_', name);


deven = csvread('./doc2vec/IWSLT17.TED.tst2015.en-fr.tok.tc.bpe.fr');
deven1 = csvread('./doc2vec/IWSLT17.TED.tst2015.en-fr.tok.tc.bped_1.fr');
deven2 = csvread('./doc2vec/IWSLT17.TED.tst2015.en-fr.tok.tc.bped_2.fr');
deven3 = csvread('./doc2vec/IWSLT17.TED.tst2015.en-fr.tok.tc.bped_3.fr');
[n, dim] = size(deven);
D = repmat([0,1,2,3], 1, n)';
data = [];
for i = 1 : n
    data = [data;deven(i,:); deven1(i,:); deven2(i,:); deven3(i,:)];
end

N = size(data, 1);

num_iter = 20;

active_indices = (1:N)';
% loss 
p = 0.2;
for iter = 1:num_iter
    [ind, scores] = filterSimple(data(active_indices, :), p, mean(data));
    active_indices = active_indices(ind);
end
j = 1;
indices = zeros(n, 1);
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
b = D(a,:);
% histogram(b);
gcf = histogram(indices);
saveas(gcf, strcat('./v2figures/',name,string(p) +'_'+string(num_iter),'.pdf'));
d=[D,s];
% save(strcat(save_file, '.mat'), 'indices', 'filtered_data', '-v7');
save(strcat(save_file, string(p) +'_'+string(num_iter)+'_index.txt'), 'indices', '-ascii');
% csvwrite(strcat(save_file, '_data'), filtered_data);