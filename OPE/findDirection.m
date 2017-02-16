function [s] = findDirection(dF, d, bound)

% Tinh s = argmin<dF,x> : sum_i |x_i|  <= bound voi i = 1...d
s = zeros(d,1);
[min_value, min_index] = min(dF);
if min_value < 0
    s(min_index) = bound;
else
    [max_value,max_index] = max(dF);
    s(max_index) = -bound;
end
