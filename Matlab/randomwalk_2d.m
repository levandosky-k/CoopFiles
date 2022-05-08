
clear;
close all;

K=10000; %number of particles
Nt=200; %number of steps
nbins=[30,30]; %number of bins
save=1;


X=zeros(K,2,Nt);

%vid = VideoWriter("./randomwalk2dmovie");
%open(vid);

%plot locations
%figure(1);
%scatter(X(:,1,1),X(:,2,1));


red=[1,0,0];
blue=[0,0,1];
colors_p = [linspace(red(1),blue(1),Nt)', linspace(red(2),blue(2),Nt)', linspace(red(3),blue(3),Nt)'];
%figure(1);
%hold on;
%scatter(X(:,1,1),X(:,2,1), [], colors_p(1,:));
%axis([-10,10,-10,10]);
%currFrame=getframe(gcf);
%writeVideo(vid, currFrame);

tic
for i=2:Nt
   r=2*pi*rand(K,1); %each particle is assigned a random number between 0 and 2pi
   jump=rand(K,1);
   dx=jump .* sin(r);
   dy=jump .* cos(r);
   X(:,1,i)=X(:,1,i-1)+dx; %update x position
   X(:,2,i)=X(:,2,i-1)+dy; %update y position
   
   %scatter(X(:,1,i),X(:,2,i),[], colors_p(i,:));
   %axis([-10,10,-10,10]);
   
   %scatter(X(:,1,i),X(:,2,i));
   %currFrame=getframe(gcf);
   %writeVideo(vid, currFrame);
end
toc

%x=X(:,1,:);
%y=X(:,2,:);
%figure(2);
%surf(x(:), y(:), T);

%close(vid);

data=zeros(Nt,nbins(1),nbins(2));
for i=1:Nt
    [n,xedges,yedges]=histcounts2(X(:,1,i),X(:,2,i), nbins);
    data(i,:,:) = n;
end

if save
    savename = "rw_2d_particles" + K + "_steps" + Nt + "_xbins" + nbins(1) + "_ybins" + nbins(2) + ".csv";
    writematrix(data, savename);
end

figure(2);
histogram2(X(:,1,Nt),X(:,2,Nt), nbins);

