%% Initialization
clear ; close all; clc;

%% =======================Setting Up
data=load('ex1data1.txt');
n=size(data,2)-1;%n=length(data(1,:)) -1; %n=number of features   n+1th col is output col
m=size(data,1);  %m=length(y); #training examples (return maximum length among all dimensions)

X=data(:,1:n);
y=data(:,n+1);
  
%Plotting data
%Before starting on any task, it is often useful to understand the data by
%visualizing it. You can always do this if dataset has only 2 properties to 
%plot. Many other problems that you will encounter in real life are 
%multi-dimensional and can't be plotted on a 2-d plot.
figure(1);
plot(X,y,'rx','Markersize',10);%for 2D graph, X=single feature ie col-matrix
xlabel('size');
ylabel('price');

%% =====================Hypothsis function
%h(X)= ?0*x0 + ?1*x1 +?2*x2+...+?n*xn    n+1 features 

%%Feature Normalization%%
%[X mu sigma] = featureNormalize(X);
%Y can also bhi normalized similarly

mu=mean(X);
sigma=std(X);   
for i=1:size(X,1)       %for ith example
    X(i,:)=(X(i,:)-mu)./sigma;
end

X=[ones(m,1), X];    %adding col for feature X0=1 
theta=zeros(n+1,1);  %initialize n+1 row matrix   

%% ===========================Cost function   (Calculate error in hypothesis)
%J(?) = 1/2m ? ( h(I)-y(i) )^2        I=ith training input
%computeCost(X,y,theta);

%Vectorized form
J=0;
J= 1/(2*m)* sum((X*theta - y).^2);         %[h()= X*theta]



%% ===========================Batch Gradient Descent  (Minimize J(?);)
fprintf('Gradient Descent Method\n');
%Non vectorized form: ?(j) = ?(j) - alpha * Derivative J(?) wrt ?j
%Vectorized form: ? = ? - alpha * Derivative J(?) wrt ?

alpha=0.03;
iterations=1000; 
%[theta,J_history]=gradientDescent(X,y,theta, alpha, iterations);

%Non vectorized form
%J_history=zeros(iterations,1);
%for k=1:iterations
%   ?=theta;
%	for j=1:n+1        %jth theta 
%       t=0;
%       for i=1:m      %ith training example
%           I=X(i,:);
%           t=t + (I*?-y(i))*X(i,j);   %derivative of ?j*(not_a_fn_of_?j) = not_a_fn_of_?j =X(i,j)
%       end
%       ?(j)=?(j)-alpha/m*t;
%   end
%   theta=?;
%   J_history(k) = computeCost(X, y, theta);
%end   

%Visualize  (Ex= ith example's Expected_value - Actual_value = X*theta-y)
%theta1=  Ex1*f1  Or f1*Ex1     
%         Ex2*f1     f1*Ex2     
%           ::         ::                           
%         Ex(m)*f1   f1*Ex(m)   
%grad 1 = f1*EX1 + f1*Ex2 +... f1*Exm
%grad 2 = f2*EX1 + f2*Ex2 +... f2*Exm
%  ::
%gradn+1= fn+1*EX1 + fn+1*Ex2 +... fn+1*Exm
%[Note 1st f1 is 1st example's f1; 2nd f1 is 2nd examples f1.]
%           X'            *        Ex
% F1 [f1          f1 ]            [Ex1]
% F2 [               ]            [Ex2]
%    [               ]            [   ]
%Fn+1[               ]            [Ex(m)]

%So grad is 1/m *  X' * Ex.

%Vectorized form
J_history=zeros(iterations,1);
for k=1:iterations
   theta=theta-alpha/m * X' *(X*theta-y);
   J_history(k) = 1/(2*m)* sum((X*theta - y).^2);%J_history(k) = computeCost(X, y, theta);
end


%% ===========================Check Working(Convergence of gradient descent)
%NOTE always J>=0 and after running algo should converge to a steady value.
%If J(?) increases, alpha needs to be decreased.
%If J(?) is very-2 slowly converging ie., J(?_initial)~J(?_final) or J(?_final)!~0 then inc alpha, or inc iterations

%Check by printing values Or
%fprintf('J history:\n ');
%for i=1:iterations   %or you could just print first 10 values
%    fprintf('%f ',J_history(i));   

%Plot the convergence graph
figure(2);
plot(1:numel(J_history), J_history,'-b','LineWidth',2);
xlabel('Number of iterations');
ylabel('Cost J');

fprintf('Theta computed from the gradient descent: \n');
fprintf(' %f \n', theta);

%% ===========================Predict
%Plotting prediction graph
figure(1);
hold on; % keep previous plot(figure(1)) visible
plot(data(:,1:n), X*theta, '-'); 
legend('Training data', 'Linear regression')
hold off; % don't overlay any more plots on this figure

% Estimate the output of 1st input example from dataset.
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.
input=data(1,1:n);
input=(input-mu)./sigma;
input=[1,input];
output=input*theta; %de-normalize output if you normalized it before

fprintf('Predicted output:\n $%f\n', output);
fprintf('Actual output:\n $%f\n', y(1));
fprintf('\n');

%% ===========================Normal Equation
fprintf('Normal Equation Method:\n');

data = csvread('ex1data1.txt');
m = size(data,1);
n = size(data,2)-1;

X = data(:, 1:n);
y = data(:, n+1);

X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = zeros(size(X, 2), 1);
%theta = normalEqn(X, y);

theta=pinv(X'*X)*X'*y;


% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);

% Estimate the output of 1st input example from dataset.
input=X(1,:);
output=input*theta;

fprintf('Predicted output:\n $%f\n', output);
fprintf('Actual output:\n $%f\n', y(1));