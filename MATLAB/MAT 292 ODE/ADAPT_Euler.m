function [sol, t_val] = ADAPT_Euler(f_prime, t0, tN, y0, h)

    %t = linspace(t0, tN, (tN-t0)/h); 
    h1 = h/2; 
    y = [y0];
    y1 = [y0];
    tol = 10^-8;
    sol = [y0];
    i = 2; %keeps track of the y index
    t = t0; 
    t_val = [t];
  
    while t < tN
        
        y = [y y(i-1) + f_prime(t, y(i-1))*h];%computes the y value with one euler step

        for x = 1:2
            if(x == 1)
                curr = y1(i-1) + f_prime(t,y1(i-1))*h1;
            else
                y1 = [y1(1:i-1) curr + f_prime(t+h1, curr) * h1];%computes the value with two euler steps of h/2
                
            end 
        end
        if (abs(y(i) - y1(i)) < tol)
            h = 0.9*h*min(max(tol/abs(y(i) - y1(i)), 0.3), 2); %update stepsize regardless of abs(D) > tol 
            h1 = h/2;
        end
        while(abs(y(i) - y1(i)) > tol)
            h = 0.9*h*min(max(tol/abs(y(i) - y1(i)), 0.3), 2); %update stepsize regardless of abs(D) > tol 
            h1 = h/2; 
            y = [y(1:i-1) y(i-1) + f_prime(t, y(i-1))*h ];
            for x = 1:2
                if(x == 1)
                    curr = y1(i-1) + f_prime(t, y1(i-1))*h1;
                else
                    y1 = [y(1:i-1) curr + f_prime(t+h1, curr) * h1];
                end 
            end           
        end
    
        sol = [sol y1(i) - y(i) + y1(i)];
        
        i = i +1; 
        t = t+h;
        t_val = [t_val t];
    end