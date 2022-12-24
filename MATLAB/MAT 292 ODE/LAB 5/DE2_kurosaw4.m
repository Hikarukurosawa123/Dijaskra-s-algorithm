function [t, y] = DEsolver(p, q, g, t0, tN, y0, y1, h)

    t = linspace(t0, tN, (tN-t0)/h +1 );
    y = zeros(size(t));
    y(1) = y0;
    y(2) = y(1) + y1*h;
    for i = 2:size(t,2)-1
        y(i+1) = 2*y(i) - y(i-1) - (y(i) - y(i-1))*h*p(t(i)) - q(t(i))*y(i)*h^2 + g(t(i))* h^2; %rearranged after plugging the derived form of y''(i), y'(i). and y(1) into the oridingal 2nd order ODE 
        
    end




end 
