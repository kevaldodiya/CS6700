function [t] = median_elim(k,times,runs)
%epsilon, delta initiliazation
t =0;
epsilon = [2.9];
delta = [0.9];
av_reward =[];
av_optimal =[];
for e = 1:size(delta,2)             %number of epsilons
    
    reward = [];
    action = [];
    arm = k;
    %taking runs * k different means from normal distribution
    
    original = normrnd(0,1,runs,k);
    count = 0;
    %finds maximum(most favourable) arm(action)
    [val,in] = max(original,[],2);
    epsilon1 = epsilon(e)/4;
    delta1 = delta(e)/2;
    l=1;
    mutotal = zeros(runs,k);
    indices = [1:k];
    t1 = indices;
    for i=1:(runs-1)  %to find optimal action percentage
        indices = [indices;t1];
    end
    while arm ~= 1
        num = ceil((4/(epsilon1^2))*log(3/delta1)); % L1, L2 computation
        for j = 1: num
            temp = normrnd(original,1);
            mutotal = mutotal + temp;
            reward = [reward,mean(temp)];
            count = count +1;
        end
        
        %for optimal action 
        for q =1:num
            temp_action = zeros(1,arm);
            for i=1:runs
                for j=1:arm
                    if(indices(i,j) == in(i))
                        temp_action(j) = temp_action(j) + 1;
                    end
                end
            end
            temp_action = temp_action/runs;
            action = [action,temp_action];
        end
        
        mutemp = mutotal/count;
        ar_med = median(mutemp,2);
        temp1 =[];
        or_temp =[];
        %taking out greter than median elements
        for i=1:runs
            jk =1;
            for j=1:arm
                if(ar_med(i) < mutemp(i,j))
                    temp1(i,jk) = mutotal(i,j);
                    or_temp(i,jk) = original(i,j);
                    t_indices(i,jk) = indices(i,j);
                    jk = jk + 1;
                end
            end
        end
        mutotal = temp1;
        original = or_temp;
        indices = t_indices;
        arm = floor(arm-arm/2);
        epsilon1 = epsilon1*0.75;
        delta1 = delta1*0.5;
        l = l+1;
    end
    num = ceil((4/(epsilon1^2))*log(3/delta1));
    num = ceil(num/2);
    for j = 1: num
        temp = normrnd(original,1);
        mutotal = mutotal + temp;
        reward = [reward,mean(temp)];
        count = count +1;
    end
    for q =1:num
            temp_action = zeros(1,arm);
            for i=1:runs
                for j=1:arm
                    if(indices(i,j) == in(i))
                        temp_action(j) = temp_action(j) + 1;
                    end
                end
            end
            temp_action = temp_action/runs;
            action = [action,temp_action];
        end
    action = action * 100;
    av_reward = reward;
    
    figure(1);
    plot(av_reward);
    hold on;
    
    figure(2);
    plot(action);
    hold on;
    t = size(action,2);
end

end