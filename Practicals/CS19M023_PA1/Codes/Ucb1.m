function [av_reward,av_optimal] = Ucb1(k,times,runs)
%initialization

mucap = zeros(1,k);
count = zeros(1,k);
av_reward =[];
av_optimal = [];
c = [1];
for e = 1:1             %number of learning rates
    reward = zeros(1,times);
    action = zeros(1,times);
    for i =1:runs
        
        %taking 10 different means from normal distribution
        
        original = normrnd(0,1,k,1);
       
        %finds maximum(most favourable) arm(action)
        [v,in] = max(original);
        rk = reward;
        mucap = zeros(1,k);
        count = zeros(1,k);
        for q = 1:k
            count(q) = count(q)+1;
            temp = normrnd(original(q),1);
            mucap(q) = temp;
            if in == q      % for optimal action graph
               action(q) = action(q) + 1;
            end
            reward(q) = temp;
        end
        for j = k+1: times
            ind = findingucb(mucap,c(e),j,count);
            if in == ind      % for optimal action graph
               action(j) = action(j) + 1;
            end
            temp = normrnd(original(ind),1);
            mucap(ind) = (mucap(ind)*count(ind) + temp)/(count(ind)+1);
            count(ind) = count(ind) + 1;
            reward(j) = temp;
            
        end
        reward = reward + rk;
    end
    disp(count);    
    rew = reward / runs;
    av_reward = [av_reward;rew];
    opt = (action / runs)*100;
    av_optimal = [av_optimal ; opt];
end
%{
figure(1);
    for i=1:4
        plot(av_reward(i,:));
        hold on
    end
    title(" UCB1 ");
    legend("C = 0.01","C = 0.1","C = 1","C = 10");
    xlabel("Times");
    ylabel("Avg Rewards");
    
    figure(2);
    for i=1:4
        plot(av_optimal(i,:));
        hold on
    end
    title(" UCB1 ");
    legend("C = 0.01","C = 0.1","C = 1","C = 10");
    xlabel("Times");
    ylabel("Optimal Actions In %");
%}
end
