% read in the iris data
x = csvread('iris.data.csv');

% training set
p = x(1:100, 1:4);
t = x(1:100, 5);

% test set 
a = x(101:150, 1:4);
s = x(101:150, 5);

% setup feed forward network
net = feedforwardnet(10);

% set the number of epochs
net.trainParam.epochs = 3000;

% set the learning rate
net.trainParam.lr = 0.3;

% set the momentum term
net.trainParam.mc = 0.6;

% normalize the data
[pn, ps] = mapminmax(p');
[tn, ts] = mapminmax(t');

[an, as] = mapminmax(a');
[sn, ss] = mapminmax(s');

% train the nn
net = train(net, pn, tn);

% simulate the test set
y = net(an);

% de-normalize y values so they return to regular values usin
% settings from the training target settings
t_test = mapminmax('reverse', y', ts);

% plot the results
% r-- means red dashed line
plot(t_test, 'r--');
hold;
plot(s);
title('Question 3 - Iris Data Comparison between actual targets and predictions');
xlabel('Datum Instance');
ylabel('Value');

% calculate the mean squared error, the lower it is the more accurate
% the training and testing data are to each other
d= (t_test-s).^2;
mse = mean(d);
