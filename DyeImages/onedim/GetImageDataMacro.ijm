for (i=7030; i <= 7030; i=i+1){
filename = "IMG_" + i;

open("/Users/katie/Desktop/CoopFiles/DyeImages/onedim2/Images/" + filename);
run("Split Channels");
selectWindow(filename);
//setTool("multipoint");

s=selectionType();
while(s==-1){
s=selectionType();
wait(1);
}
getSelectionCoordinates(xPoints, yPoints);
x = xPoints[0];
y = yPoints[0];

makeLine(x-600, y, x+600, y);

run("Plots...", "width=1000 height=340 font=14 draw_ticks list minimum=0 maximum=0 interpolate");
run("Plot Profile");
selectWindow("Plot Values");
saveAs("Results", "/Users/katie/Desktop/CoopFiles/DyeImages/onedim2/PlotValues" + i + ".csv");

selectWindow(filename);
close();
selectWindow("Plot of " + "IMG_" + i);
close();
selectWindow("PlotValues" + i + ".csv");
close("PlotValues" + i + ".csv");
}

