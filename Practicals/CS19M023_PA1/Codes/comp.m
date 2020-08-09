%initialization
tic
k = input("number of arms");
times = input("times");
runs = input("runs");
k=100;
[t] = median_elim(k,times,runs);
k =1000;
[eps_gr_reward,eps_gr_optimal] = eps1(k,times,runs);
[Ucb1_reward,Ucb1_optimal] = Ucb1(k,times,runs);
[Softmax_reward,Softmax_optimal] = Softmax(k,times,runs);
av_reward = [];
av_reward = [av_reward;eps_gr_reward];
av_reward = [av_reward;Ucb1_reward];
av_reward = [av_reward;Softmax_reward];
av_optimal = [];
av_optimal = [av_optimal;eps_gr_optimal];
av_optimal = [av_optimal;Ucb1_optimal];
av_optimal = [av_optimal;Softmax_optimal];

for i=1:3
    figure(1);
    plot(av_reward(i,:));
    hold on
end
title(" Comparison ");
legend("eps = 1.2  del = 0.8","eps(eps-greedy) = 0.1","temp(softmax) = 0.3","c(UCB1) = 1");
xlabel("Times");
ylabel("Avg Rewards");


figure(2);
for i=1:3
    plot(av_optimal(i,:));
    hold on
end
title(" Comparison ");
legend("eps = 1.2  del = 0.8","eps(eps-greedy) = 0.1","temp(softmax) = 0.3","c(UCB1) = 1");
xlabel("Times");
ylabel("Optimal Actions In %");

disp(toc);