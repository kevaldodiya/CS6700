function [av_reward,av_optimal] = eps1(k,times,runs)
%initialization

mucap = zeros(1,k);
count = zeros(1,k);
av_reward =[];
av_optimal = [];
for e = 1:1             %number of epsilons
    reward = zeros(1,times);
    eps =0.1;
    action = zeros(1,times);
    for i =1:runs
        
        %taking k different means from normal distribution
        
        original = normrnd(0,1,k,1);
       
        %finds maximum(most favourable) arm(action)
        [v,in] = max(original);
        rk = reward;
        mucap = zeros(1,k);
        count = zeros(1,k);
        for j = 1: times
            r = rand;
            if(r > eps)
                
                %find arm which has best abg mean till now
                [val,ind] = max(mucap);
                if in == ind      % for optimal action graph
                    action(j) = action(j) + 1;
                end
                temp = normrnd(original(ind),1);
                % adding new reward to previous avg reward
                mucap(ind) = (mucap(ind)*count(ind) + temp)/(count(ind)+1);
                count(ind) = count(ind) + 1;
                %mucap(ind) = mucap(ind)+ (1/count(ind))*(temp-mucap(ind));
                reward(j) = temp;
            else
                
                % epsilon time random selection
                q = randi(10);
                
                if in == q
                    action(j) = action(j) + 1;
                end
                
                temp = normrnd(original(q),1);
                mucap(q) = (mucap(q)*count(q) + temp)/(count(q)+1);
                count(q) = count(q) + 1;
                %this method uses sliding average
                %mucap(q) = mucap(q)+ (1/count(q))*(temp-mucap(q));
                reward(j) = temp;
            end
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
    for i=1:5
        plot(av_reward(i,:));
        hold on
    end
    title(" Epsilon- Greedy ");
    legend("eps = 0.1","eps = 0.01","eps = 0","eps = 1","eps = 0.02");
    xlabel("Times");
    ylabel("Avg Rewards");
    
    figure(2);
    for i=1:5
        plot(av_optimal(i,:));
        hold on
    end
    title(" Epsilon- Greedy ");
    legend("eps = 0.1","eps = 0.01","eps = 0","eps = 1","eps = 0.02");
    xlabel("Times");
    ylabel("Optimal Actions In %");
  %}  
end

