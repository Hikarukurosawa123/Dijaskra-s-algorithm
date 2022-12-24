
function y = euler2(f_prime, t0, tN, y0, h)
t = linspace(t0, tN, (tN-t0)/h+1)
y = zeros(size(t));
y(1) = y0; 
for i = 2: size(t,2)
    y(i) = y(i-1) + f_prime(t(i-1), y(i-1))*h;
end
