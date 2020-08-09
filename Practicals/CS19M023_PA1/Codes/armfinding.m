function [arm] = armfinding(mucap,beta)
    count =0;
    a =[];
    for i=1:size(mucap,2)
        temp = mucap(i);
        temp = temp / beta;
        count = count + exp(temp);
        a = [a,exp(temp)];
    end
    prob =[];
    for i=1:size(mucap,2)
        prob = [prob,a(i)/count];
    end
    arm = randsample(size(mucap,2),1,true,prob);
    
    