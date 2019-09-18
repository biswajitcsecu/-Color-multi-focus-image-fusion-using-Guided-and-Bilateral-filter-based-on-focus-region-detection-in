function DWTBilateralGuidedFilterImageFusion()

clear; clc; close all; warning off;

%load first image
ar=imread('1.jpg');
[nr,nc, nch]=size(ar);
a = reshape(ar,[nr,nc*nch]);
a = double(a);
a = a / 255;

%load second image
br=imread('2.jpg');
[nr,nc, nch]=size(ar);
b = reshape(br,[nr,nc*nch]);
b = double(b);
b = b / 255;

%DWT transform
[a1,b1,c1,d1]=dwt2(a,'db4','mode','symw');
[a2,b2,c2,d2]=dwt2(b,'db4','mode','symw');

% wname = 'db4';
% [Lo_D,Hi_D,Lo_R,Hi_R] = wfilters(wname);
% subplot(221); stem(Lo_D);
% title('Decomposition low-pass filter');
% subplot(222); stem(Hi_D);
% title('Decomposition high-pass filter');
% subplot(223); stem(Lo_R);
% title('Reconstruction low-pass filter');
% subplot(224); stem(Hi_R);
% title('Reconstruction high-pass filter');
% xlabel('The four filters for db5')

[k1,k2]=size(a1);

%fusion map for lowpass
a3= zeros(k1,k2);
for i=1:k1
    for j=1:k2
        a3(i,j)=(a1(i,j)+a2(i,j))/2;
    end
end

%Guided parameters and for highpass image 1
% sd = sqrt(.001);
win_size1 = 3;
sigmad = 3 ;
sigmar =  0.1;
%bileteral filter
db1 = bifilter2d(b1,win_size1,sigmad,sigmar);
dc1 = bifilter2d(c1,win_size1,sigmad,sigmar);
dd1 = bifilter2d(d1,win_size1,sigmad,sigmar);

%different cof
b1= abs(b1-db1);
c1= abs(c1-dc1);
d1= abs(d1-dd1);

% inputa = imnoise(a, 'gaussian', 0, sd*sd);
guidea = a1;
bf1 = guidedFilter(b1, guidea, .001, win_size1);
cf1 = guidedFilter(c1, guidea, .001, win_size1);
df1 = guidedFilter(d1, guidea, .001, win_size1);


%for high image 2
% inputb= imnoise(b, 'gaussian', 0, sd*sd);
win_size2 = 3;

%bileteral filter
db2 = bifilter2d(b2,win_size2,sigmad,sigmar);
dc2 = bifilter2d(c2,win_size2,sigmad,sigmar);
dd2 = bifilter2d(d2,win_size2,sigmad,sigmar);

%different cof
b2= abs(b2-db2);
c2= abs(c2-dc2);
d2= abs(d2-dd2);
%guided for image2
guideb = b1;
bf2 = guidedFilter(b2, guideb, .001, win_size2);
cf2 = guidedFilter(c2, guideb, .001, win_size2);
df2 = guidedFilter(d2, guideb, .001, win_size2);

%fusion map for high-pass
b3= zeros();
c3= zeros();
d3= zeros();
for i=1:k1
    for j=1:k2
        b3(i,j)=max(bf1(i,j),bf2(i,j));
        c3(i,j)=max(cf1(i,j),cf2(i,j));
        d3(i,j)=max(df1(i,j),df2(i,j));
    end
end

%IDWT
c=idwt2(a3,b3,c3,d3,'db4','mode','symw');

%display
figure,imshow(ar)
title('First Image');

figure,imshow(br)
title('Second Image');

%fused image
c = reshape(c,[nr,nc, nch]);
figure,imshow(c,[]);
title('Fused Image');
drawnow;
end


function filtered = guidedFilter(input, guide, epsilon, win_size)

np = win_size * win_size;
half = floor(win_size / 2);

paddedp = padarray(input, [half, half], 'both');
mup = zeros(size(paddedp));

paddedi = padarray(guide, [half, half], 'both');
mui = zeros(size(paddedi));

sigmai = zeros(size(paddedi));
cross = zeros(size(paddedi));

initial_denom = padarray(ones(size(input)), [half, half], 'both');
denom = zeros(size(paddedi));

for i = -half : half
    for j = -half : half
        mup = mup + circshift(paddedp, [i, j]);
        mui = mui + circshift(paddedi, [i, j]);
        sigmai = sigmai + circshift(paddedi, [i, j]).^2;
        cross = cross + circshift(paddedi, [i, j]).*circshift(paddedp, [i, j]);
        denom = denom + circshift(initial_denom, [i, j]);
    end
end

mup = mup(half+1:end-half, half+1:end-half);
mui = mui(half+1:end-half, half+1:end-half);
sigmai = sigmai(half+1:end-half, half+1:end-half);
cross = cross(half+1:end-half, half+1:end-half);
denom = denom(half+1:end-half, half+1:end-half);

mup = mup ./ denom;
mui = mui ./ denom;
sigmai = sigmai ./ denom - mui.^2/sqrt(np);
cross = cross ./ denom;

a = (cross - mui .* mup) ./ (sigmai + epsilon);
b = mup - a .* mui;

apad = padarray(a, [half, half], 'both');
bpad = padarray(b, [half, half], 'both');

mua = zeros(size(apad));
mub = zeros(size(bpad));

for i = -half : half
    for j = -half : half
        mua = mua + circshift(apad, [i, j]);
        mub = mub + circshift(bpad, [i, j]);
    end
end

mua = mua(half+1:end-half, half+1:end-half);
mub = mub(half+1:end-half, half+1:end-half);
mua = mua ./ denom;
mub = mub ./ denom;

filtered = mua .* input + mub;
return;
end



function B = bifilter2d(A,w,sigma_d,sigma_r)

[X,Y] = meshgrid(-w:w,-w:w);
G = exp(-(X.^2+Y.^2)./(2.*sigma_d.^2));

dim = size(A);
B = zeros(dim);
for i = 1:dim(1)
    for j = 1:dim(2)
        iMin = max(i-w,1);
        iMax = min(i+w,dim(1));
        jMin = max(j-w,1);
        jMax = min(j+w,dim(2));
        I = A(iMin:iMax,jMin:jMax);
        H = exp(-(I-A(i,j)).^2/(2*sigma_r^2));
        F = H.*G((iMin:iMax)-i+w+1,(jMin:jMax)-j+w+1);
        B(i,j) = sum(F(:).*I(:))/sum(F(:));
    end
    
end

end










