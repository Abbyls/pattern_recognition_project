clear;
figure,subplot(1,2,1);
imshow('1.BMP');
title('original image')
[xx,yy,button]=ginput;
RGB=imread('1.BMP');
RR=RGB(:,:,1);GG=RGB(:,:,2);BB=RGB(:,:,3);
[count,asd]=size(xx);
subplot(1,2,2); hold on;

for i=1:count
    row = floor(yy(i)); col = floor(xx(i));
    trnx(i,:)=[RR(row,col),GG(row, col)];
    switch button(i)
        case 1
            trny(i,:)= 1;
            plot(trnx(i,1),trnx(i,2),'r*');
        case 3
            trny(i,:)= -1;
            plot(trnx(i,1),trnx(i,2),'b+');
        otherwise
    end
end
xlabel('R')
ylabel('G')

%% SVM to calculate result
ker='linear';
trnx = double(trnx);                    % change data type to double?
[nsv alpha bias]=svc(trnx, trny, ker,10);
w=sum(diag(trny.*alpha)*trnx)';
fprintf('w:        : [%f %f]\n',w(1),w(2));

w1=w(1);w2=w(2);
px=1:1:255;
py=-w1/w2.*px-bias/w2;
plot(px,py)

% figure; svcplot(trnx, trny, ker, alpha, bias);

%% Show result image

% Using matrix is fast
st=cputime;
[row,col]=size(RR);
RR1=double(reshape(RR,1,[]));
GG1=double(reshape(GG,1,[]));
bias1=ones(1,row*col)*bias;
testx=[RR1;GG1];

sigg=sign(w'*testx+bias1);
sigg=(sigg+1)*255/2;

reslm=reshape(sigg,row,col);
fprintf('Dual testing time: %4.9f seconds\n',cputime - st);

%figure;
%imshow(reslm);

%find((reslm-reslm2)>0)

%% Quadratic programming to calculate result
trnx = double(trnx);                    % change data type to double
trnx2 = trnx;
trnx2(:,3)=1;

H = [eye(2),[0;0];
    zeros(1,3)];
f=zeros(3,1);
A=diag(-trny)*trnx2;
b=-ones(count,1);

st = cputime;
result = quadprog(H,f,A,b);
fprintf('Original training time: %4.9f seconds\n',cputime - st);

w=result(1:2);
bias=result(3);
w2 = w'*w;
fprintf('|w0|^2    : %f\n',w2);
fprintf('Margin    : %f\n',2/sqrt(w2));
fprintf('w:        : [%f %f]\n',w(1),w(2));

w1=w(1);w2=w(2);
px=1:1:255;
py=-w1/w2.*px-bias/w2;
plot(px,py,'r')
hold off;


%% Show result image

% Using matrix is fast
st=cputime;
[row,col]=size(RR);
RR1=double(reshape(RR,1,[]));
GG1=double(reshape(GG,1,[]));
testx=[RR1;GG1];
bias1=ones(1,row*col)*bias;

sigg=sign(w'*testx+bias1);
sigg=(sigg+1)*255/2;

reslm2=reshape(sigg,row,col);
fprintf('Original testing time: %4.9f seconds\n',cputime - st);

figure;
imshow(reslm);
figure;
imshow(reslm2);


%% sample data
% xx=  [162.9492;
%   157.6230;
%   159.3984;
%   166.5000;
%   164.7246;
%   154.0722;
%   159.3984;
%   159.3984;
%   155.8476;
%   157.6230;
%   292.5535;
%   303.2059;
%   303.2059;
%   297.8797;
%   269.4733;
%   216.2112;
%   193.1310;
%   104.3610;
%    43.9973;
%    19.1417];
% yy=[  114.8369;
%   118.3877;
%   116.6123;
%   114.8369;
%   120.1631;
%   125.4893;
%   116.6123;
%   120.1631;
%   116.6123;
%   121.9385;
%   155.6711;
%    89.9813;
%    40.2701;
%    27.8422;
%    29.6176;
%    24.2914;
%    24.2914;
%    24.2914;
%    24.2914;
%     8.3128];
% button=[     1;
%      1;
%      1;
%      1;
%      1;
%      1;
%      1;
%      1;
%      1;
%      1;
%      3;
%      3;
%      3;
%      3;
%      3;
%      3;
%      3;
%      3;
%      3;
%      3];