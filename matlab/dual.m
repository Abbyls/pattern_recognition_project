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

%% SVM to calculate result
ker='linear';
trnx = double(trnx);                    % change data type to double?
[nsv alpha bias]=svc(trnx, trny, ker,10);
w=sum(diag(trny.*alpha)*trnx)';

figure; svcplot(trnx, trny, ker, alpha, bias);

%% Show result image

% Using 'for' is not fast
% for i=1:size(RGB,1)
%     for j=1:size(RGB,2)
%         testx=double(RGB(i,j,1:2));
%         
%         preY=svcoutput(trnx,trny,testx,ker,alpha,bias);
%         if(preY==1)
%             reslm(i,j)=255;
%         else
%             reslm(i,j)=0;
%         end
%     end
% end

% Using matrix is fast
[row,col]=size(RR);
RR1=double(reshape(RR,1,[]));
GG1=double(reshape(GG,1,[]));
testx=[RR1;GG1];
bias1=ones(1,row*col)*bias;

sigg=sign(w'*testx+bias1);
sigg=(sigg+1)*255/2;

reslm=reshape(sigg,row,col);

figure;
imshow(reslm);