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
hold off

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
fprintf('Execution time: %4.9f seconds\n',cputime - st);

w=result(1:2)
bias=result(3);
w2 = w'*w;
fprintf('|w0|^2    : %f\n',w2);
fprintf('Margin    : %f\n',2/sqrt(w2));

% Using for is not fast
% for i=1:size(RGB,1)
%     for j=1:size(RGB,2)
%         testx=[RGB(i,j,1) ; RGB(i,j,2)];
%         testx = double(testx);
%         
%         preY= w'*testx + bias;
%         if(preY>0)
%             reslm(i,j)=255;
%         else
%             reslm(i,j)=0;
%         end
%     end
% end
% 
% figure;
% imshow(reslm);


% Using matrix is fast
[row,col]=size(RR);
RR1=double(reshape(RR,1,[]));
GG1=double(reshape(GG,1,[]));
testx=[RR1;GG1];
bias1=ones(1,row*col)*bias;

sigg=sign(w'*testx+bias1);
sigg=(sigg+1)*255/2;

reslm2=reshape(sigg,row,col);

figure;
imshow(reslm2);