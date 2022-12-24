%% ODE Lab: Creating your own ODE solver in MATLAB
% In this lab, you will write your own ODE solver for the Improved Euler method 
% (also known as the Heun method), and compare its results to those of |ode45|.
% 
% You will also learn how to write a function in a separate m-file and execute 
% it.
% 
% Opening the m-file lab3.m in the MATLAB editor, step through each part using 
% cell mode to see the results. Compare the output with the PDF, which was generated 
% from this m-file.
% 
% There are six (6) exercises in this lab that are to be handed in on the due 
% date. Write your solutions in the template, including appropriate descriptions 
% in each step. Save the .m files and submit them online on Quercus.
%% Student Information
% Student Name: Hikaru Kurosawa
% 
% Student Number: 1007675240
%% Creating new functions using m-files.
% Create a new function in a separate m-file:
% 
% Specifics: Create a text file with the file name f.m with the following lines 
% of code (text):
%%
% 
%  function y = f(a,b,c)
%  y = a+b+c;
%
%% 
% Now MATLAB can call the new function f (which simply accepts 3 numbers and 
% adds them together). To see how this works, type the following in the matlab 
% command window: sum = f(1,2,3)
% 

%% Exercise 1
% Objective: Write your own ODE solver (using the Heun/Improved Euler Method).
% 
% Details: This m-file should be a function which accepts as variables (t0,tN,y0,h), 
% where t0 and tN are the start and end points of the interval on which to solve 
% the ODE, y0 is the initial condition of the ODE, and h is the stepsize. You 
% may also want to pass the function into the ODE the way |ode45| does (check 
% lab 2).
% 
% Note: you will need to use a loop to do this exercise. You will also need 
% to recall the Heun/Improved Euler algorithm learned in lectures.


f = @(t,y)t;
sol = @(t) t.^2/2;
%sam
y = Improved_EULER(f, 0, 0.8, 0, 0.025)


%Note: Code written at bottom
%% Exercise 2
% Objective: Compare Heun with |ode45|.
% 
% Specifics: For the following initial-value problems (from lab 2, exercises 
% 1, 4-6), approximate the solutions with your function from exercise 1 (Improved 
% Euler Method). Plot the graphs of your Improved Euler Approximation with the 
% |ode45| approximation.
% 
% (a) |y' = y tan t + sin t, y(0) = -1/2| from |t = 0| to |t = pi|
% 
% (b) |y' = 1 / y^2 , y(1) = 1| from |t=1| to |t=10|
% 
% (c) |y' = 1 - t y / 2, y(0) = -1| from |t=0| to |t=10|
% 
% (d) |y' = y^3 - t^2, y(0) = 1| from |t=0| to |t=1|
% 
% Comment on any major differences, or the lack thereof. You do not need to 
% reproduce all the code here. Simply make note of any differences for each of 
% the four IVPs.

y_pa = @(t,y)y.*tan(t) + sin(t);
y_pb =  @(t,y)1 / y.^2;
y_pc =  @(t,y)1 - t*y./2;
y_pd =  @(t,y)y.^3 - t.^2;

t0 = 0; 
tN = pi; 
h = 0.1;
y0 = -1/2; 
%a
y_a = Improved_EULER(y_pa, t0, tN, -1/2, 0.1);
t = linspace(t0, tN, (tN-t0)/h+1); 

soln = ode45(y_pa, [t0, tN], -1/2);
plot(t, y_a, soln.x, soln.y);
legend("improved", "ode45")
ylim([-10, 10])
xlim([t0,tN])
%a) while the ode approximation produces a smooth nondecreasing curve, the euler method
%produces an increasing graph that has a huge downward spike at right
%before t = 1.5. After this point, the improved euler method approximation
%takes a totally different curve trajectory. 


%(b) y' = 1 / y^2 , y(1) = 1 from t=1 to t=10, 
% similar curve trajectory 
t0 = 1; 
tN = 10; 
h = 0.1;
y0 = 1; 
y_a = Improved_EULER(y_pb, t0, tN, y0, 0.1);
t = linspace(t0, tN, (tN-t0)/h+1); 


soln = ode45(y_pb, [t0, tN], y0);
plot(t, y_a, soln.x, soln.y);
legend("improved", "ode45")
ylim([-10, 10])
xlim([t0,tN])

%(c) y' = 1 - t y / 2, y(0) = -1 from t=0 to t=10
%the curves are roughly similar, but the two approximations start to
%differ at around t = 0.5. The improved approximation eventually shows a
%decreasing function while the ode45 approximation shows an increasing
%curve.
t0 = 0; 
tN = 10; 
h = 0.1;
y0 = -1; 
y_a = Improved_EULER(y_pc, t0, tN, y0, 0.1);
t = linspace(t0, tN, (tN-t0)/h+1); 
soln = ode45(y_pb, [t0, tN], y0);
plot(t, y_a, soln.x, soln.y);
legend("improved", "ode45")
ylim([-10, 10])
xlim([t0,tN])

%(d) y' = y^3 - t^2, y(0) = 1 from t=0 to t=1
%similar trajectory until t = 0.5 but the euler approximation starts to diverge upwards from this point and onwards.

t0 = 0; 
tN = 1; 
h = 0.1;
y0 = 1; 
y_a = Improved_EULER(y_pd, t0, tN, y0, 0.1);
t = linspace(t0, tN, (tN-t0)/h+1); 

soln = ode45(y_pb, [t0, tN], y0);
plot(t, y_a, soln.x, soln.y);
legend("improved", "ode45")
ylim([-10, 10])
xlim([t0,tN])
%% Exercise 3
% Objective: Use Euler's method and verify an estimate for the global error.
% 
% Details:
% 
% (a) Use Euler's method (you can use euler.m from iode) to solve the IVP
% 
% |y' = 2 t sqrt( 1 - y^2 ) , y(0) = 0|
% 
% from |t=0| to |t=0.5|.
% 
% (b) Calculate the solution of the IVP and evaluate it at |t=0.5|.
% 
% (c) Read the attached derivation of an estimate of the global error for Euler's 
% method. Type out the resulting bound for En here in a comment. Define each variable.
% 
% (d) Compute the error estimate for |t=0.5| and compare with the actual error.
% 
% (e) Change the time step and compare the new error estimate with the actual 
% error. Comment on how it confirms the order of Euler's method.

%a
f = @(t,y) 2.*t.*sqrt(1-y.^2);
y = euler2(f, 0, 0.5, 0, 0.01);

%euler 2 function code below
%%
%by seperation of variables
% 1) 1/(1-y^2)^(1/2) dy = 2t dt 
% 2) arcsin(y) = t^2 + C
% 3) y = sin(t^2 +C) -> genral solution 
% 4) for y(0) = 0, C = 0
% 5) particular solution = y = sin(t^2)
sol = @(t) sin(t^2);
y_real = sol(0.5) %real solution: y(0.5) = 0.2474

%c)
h = 0.01;
%M (the maximum bound) is computed by finding the maximum absolute values among the
%computed values of the following functions at t = 0.5: 
% (1) df/dy = -2ty/(1-y^2)^1/2, (2) df/dt = 2*(1-y^2)^1/2, (3) f = 2t(1-y^2)^1/2

% (2) setting t = 0.5 as the maximum value and y = 0.2474 (because sin is
% an increasing function in the range (0, pi)) as the y max value, |M| >=
% 1.9378

% (3) setting t = 0.5 as the maximum value and y = 0.2474 (because sin is
% an increasing function in the range (0, pi)) as the y max value, |M| >=
% 0.9689

%although eqn (1) is not bounded, the bound for M^2 can be found by 
%df/dy * f, which equates to -4t^2y. Using t = 0.5 and y = 0.2474, M^2 >=
%0.2474. Therefore, M >= 0.4974. 


%En <= (1+1.9378)*0.1/2*(exp(1.9378*h*n)-1) for M = 1.9378, h is the step
%size and n is the number of iterations in which the error is computed 
t = linspace(0, 0.5, (0.5-0)/h+1); 
M = 1.9378;
%d) 
err_est = zeros(1,0.5/h+1);
err_est (1) = 0; 

for i = 2:0.5/h+1
    err_est(i) = (1+h*M)*abs(sol(t(i-1))-y(i-1)) + (h)^2/2*(M+M^2);
end


y_err = sol(0.5)- y(size(t,2)) % actual error = 0.0047
err_est(0.5/h+1) % error estimate = 0.0050

h = 0.001
y = euler2(f, 0, 0.5, 0, h);
t = linspace(0, 0.5, (0.5-0)/h+1); 

err_est = zeros(1,0.5/h+1);
for i = 2: 0.5/h+1
    err_est(i) = (1+h*M)*abs(sol(t(i-1))-y(i-1)) + (h)^2/2*(M+M^2);
end 
y_err = sol(0.5)- y(0.5/h+1) % actual error = 4.7230*10^-4

err_est(0.5/h+1) % error estimate = 4.7534*10^-4


h = 0.0001
y = euler2(f, 0, 0.5, 0, h);
t = linspace(0, 0.5, (0.5-0)/h+1); 

err_est = zeros(1,0.5/h+1);
for i = 2: 0.5/h+1
    err_est(i) = (1+h*M)*abs(sol(t(i-1))-y(i-1)) + (h)^2/2*(M+M^2);
end 
y_err = sol(0.5)- y(0.5/h+1)% actual error =4.7221*10^-5

err_est(0.5/h+1) % error estimate = 4.7252*10^-5
%since the global error decreases by a factor of ten as the step size
%decreases by a factor of ten, it confirms that euler's method is a
%first-order method. 
%% Adaptive Step Size
% As mentioned in lab 2, the step size in |ode45| is adapted to a specific error 
% tolerance.
% 
% The idea of adaptive step size is to change the step size |h| to a smaller 
% number whenever the derivative of the solution changes quickly. This is done 
% by evaluating f(t,y) and checking how it changes from one iteration to the next.
%% Exercise 4
% Objective: Create an Adaptive Euler method, with an adaptive step size |h|.
% 
% Details: Create an m-file which accepts the variables |(t0,tN,y0,h)|, as in 
% exercise 1, where |h| is an initial step size. You may also want to pass the 
% function into the ODE the way |ode45| does.
% 
% Create an implementation of Euler's method by modifying your solution to exercise 
% 1. Change it to include the following:
% 
% (a) On each timestep, make two estimates of the value of the solution at the 
% end of the timestep: |Y| from one Euler step of size |h| and |Z| from two successive 
% Euler steps of size |h/2|. The difference in these two values is an estimate 
% for the error.
% 
% (b) Let |tol=1e-8| and |D=Z-Y|. If |abs(D)<tol|, declare the step to be successful 
% and set the new solution value to be |Z+D|. This value has local error |O(h^3)|. 
% If |abs(D)>=tol|, reject this step and repeat it with a new step size, from 
% (c).
% 
% (c) Update the step size as |h = 0.9*h*min(max(tol/abs(D),0.3),2)|.
% 
% Comment on what the formula for updating the step size is attempting to achieve.


f = @(t,y) t;
sol = ADAPT_Euler(f, 0, 0.5, 0, 0.01);
plot(sol)


%the formula is essentially trying to make the error value over the entire
%approximation process roughly constant. This is accomplsihed by decreasing
%the stepsize when the error is high, and increasing the step size when the
%error is smaller than the threshold error value defined by tol. It
%evaluates the higher value between tol/abs(y(i)-y1(i)) and 0.3, in which
%the former becomes smaller only when abs(y(i)-y1(i)) > tol/0.3. Then the
%evaluated value is again compared with 2, and the smaller value is used as
%the step size. This process ensures that step size does not get neither
%too small nor too big. 
%% 
% 
%% Exercise 5
% Objective: Compare Euler to your Adaptive Euler method.
% 
% Details: Consider the IVP from exercise 3.
% 
% (a) Use Euler method to approximate the solution from |t=0| to |t=0.75| with 
% |h=0.025|.
% 
% (b) Use your Adaptive Euler method to approximate the solution from |t=0| 
% to |t=0.75| with initial |h=0.025|.
% 
% (c) Plot both approximations together with the exact solution.

f = @(t,y) 2.*t.*sqrt(1-y.^2);
y = euler2(f, 0, 0.75, 0, 0.025);
sol = @(t) sin(t.^2);

[y1, t_val] = ADAPT_Euler(f, 0, 0.75, 0, 0.025);
t = linspace(0, 0.75, 0.75/0.025+1)

plot(t,y, t_val, y1, t, sol(t));
legend("euler", "adapt", "real");
%% Exercise 6
% Objective: Problems with Numerical Methods.
% 
% Details: Consider the IVP from exercise 3 (and 5).
% 
% (a) From the two approximations calculated in exercise 5, which one is closer 
% to the actual solution (done in 3.b)? Explain why.

% The Euler approximation is closer to the actual solution than the adaptive euler method. This is because the derivative value depends on the y value approximated from the previous iteration, and the y value it self has errors accumulated as more approximations are made. As a result, both the error values for the derivative and y value becomes non negligible as more steps are made. Therefore, the euler method is more accurate sincre it uses less values that has less accumulation of errors.
%% 
% (b) Plot the exact solution (from exercise 3.b), the Euler's approximation 
% (from exercise 3.a) and the adaptive Euler's approximation (from exercise 5) 
% from |t=0| to |t=1.5|.

f = @(t,y) 2.*t.*sqrt(1-y.^2);
y = euler2(f, 0, 1.5, 0, 0.025);
sol = @(t) sin(t.^2); %for 

[y1, t_val] = ADAPT_Euler(f, 0, 1.5, 0, 0.025);
t = linspace(0, 1.5, 1.5/0.025+1)

plot(t,y, t_val, y1, t, sol(t));
legend("euler", "adapt", "real")
%% 
% (c) Notice how the exact solution and the approximations become very different. 
% Why is that? Write your answer as a comment.

%Since the values of the solutions are approximated using the derivative
%2t(1-y^2)^(1/2), this function has a range of (-infinity, 1] for y,
%where the inside of the square root remains non negative. After y becomes
%greater than 1, the derivative essentially becomes zero, so the
%approximation of the curve shows a flat line from that point and onwards.
%Therefore, the actual solution is a piecewise function 
% f = {sin(t^2) for 0<t<(pi/2)^1/2 
%     {1 for (pi/2)^1/2 < t < inifinity
%%




