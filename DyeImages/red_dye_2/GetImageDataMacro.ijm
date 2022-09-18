for (i=3; i <= 150; i=i+5){
j=i+300;
if (i<10){
filename = "ezgif-frame-00" + i + ".png";
}
else if (i<100){
filename = "ezgif-frame-0" + i + ".png";
}
else{
filename = "ezgif-frame-" + i + ".png";
}

open("/Users/katie/Desktop/CoopFiles/DyeImages/set2/images3/" + filename);
run("Split Channels");
selectWindow(filename + " (green)");
close();
selectWindow(filename + " (blue)");
close();
selectWindow(filename + " (red)");
//setTool("multipoint");


s=selectionType();
while(s==-1){
s=selectionType();
wait(1);
}
getSelectionCoordinates(xPoints, yPoints);
x = xPoints[0];
y = yPoints[0];

makeLine(x-150, y, x+150, y);

run("Plots...", "width=1000 height=340 font=14 draw_ticks list minimum=0 maximum=0 interpolate");
run("Plot Profile");
selectWindow("Plot Values");
saveAs("Results", "/Users/katie/Desktop/CoopFiles/DyeImages/set2/PlotValues" + j + ".csv");

selectWindow(filename + " (red)");
close();
selectWindow("Plot of " + filename + " (red)");
close();
selectWindow("PlotValues" + j + ".csv");
close("PlotValues" + j + ".csv");
}

