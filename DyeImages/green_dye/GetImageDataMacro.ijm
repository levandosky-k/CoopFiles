for (i=6211; i <= 6249; i=i+1){
filename = "IMG_" + i + ".JPG";

open("/Users/katie/Desktop/CoopFiles/DyeImages/set3/Images/" + filename);
run("Split Channels");
selectWindow(filename + " (red)");
close();
selectWindow(filename + " (blue)");
close();
selectWindow(filename + " (green)");
//setTool("multipoint");

s=selectionType();
while(s==-1){
s=selectionType();
wait(1);
}
getSelectionCoordinates(xPoints, yPoints);
x = xPoints[0];
y = yPoints[0];

makeLine(x-400, y, x+400, y);

run("Plots...", "width=1000 height=340 font=14 draw_ticks list minimum=0 maximum=0 interpolate");
run("Plot Profile");
selectWindow("Plot Values");
saveAs("Results", "/Users/katie/Desktop/CoopFiles/DyeImages/set3/PlotValuesb" + i + ".csv");

selectWindow(filename + " (green)");
close();
selectWindow("Plot of " + filename + " (green)");
close();
selectWindow("PlotValuesb" + i + ".csv");
close("PlotValuesb" + i + ".csv");
}

