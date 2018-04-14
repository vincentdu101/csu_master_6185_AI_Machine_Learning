x = csvread('iris.data.csv');

% training set
p = x(1:100, 1:4);
T = x(1:100, 5);

% test set 
t = x(101:150, 1:4);
tT = x(101:150, 5);

% setup feed forward network
net = feedforwardnet(10);

% set the number of epochs
net.trainParam.epochs = 30;

% set the learning rate
net.trainParam.lr = 0.3;

% set the momentum term
net.trainParam.mc = 0.6;

% train the nn
[pn, ps] = mapminmax(p');
[Tn, Ts] = mapminmax(T');

[tn, ts] = mapminmax(t');
[tTn, tTs] = mapminmax(tT');

net = train(net, pn, Tn);
y = net(tn);
t_test = mapminmax('reverse', y', ts);

% plot the results
% r-- means red dashed line
plot(t_test, 'r--');
hold;
plot(T);
title('Comparison between actual targets and predictions');
xlabel('Datum Instance');
ylabel('Value');

d= (t_test-T).^2;
mse = mean(d);
