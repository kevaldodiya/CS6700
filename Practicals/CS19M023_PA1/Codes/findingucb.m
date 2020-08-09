function [ind] = findingucb(mucap,c,t,count)
    upper = zeros(1,size(mucap,2));
    for i=1:size(mucap,2)
        k = log(t)/count(i);
        temp = mucap(i) + c*sqrt(k) ;
        upper(i) = temp;
    end
    [value,ind] = max(upper);
end