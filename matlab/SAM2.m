function [ sam] = SAM2( a1, a2 )
[m,n,l]=size(a1);

k=0;
a1=Unfold(a1,size(a1),3);
a2=Unfold(a2,size(a2),3);
b=any(a1);
a1=a1(:,b);
a2=a2(:,b);
c=any(a2);
a1=a1(:,c);
a2=a2(:,c);
a1=normc(a1);
a2=normc(a2);
a3=a1.*a2;
% a3=abs(a3);
a3=sum(a3);
sam=mean(rad2deg(acos(a3)));