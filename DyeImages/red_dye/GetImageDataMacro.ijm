for (i=1; i <= 18; i++){
filename = "dye" + i + ".png";
open("/Users/katie/Desktop/CoopFiles/DyeImages/" + filename);
run("Split Channels");
selectWindow("dye" + i + ".png (green)");
close();
selectWindow("dye" + i + ".png (blue)");
close();
selectWindow("dye" + i + ".png (red)");
//setTool("multipoint");


s=selectionType();
while(s==-1){
s=selectionType();
wait(1);
}
getSelectionCoordinates(xPoints, yPoints);
x = xPoints[0];
y = yPoints[0];

makeLine(x-75, y, x+75, y);

run("Plots...", "width=1000 height=340 font=14 draw_ticks list minimum=0 maximum=0 interpolate");
run("Plot Profile");
selectWindow("Plot Values");
saveAs("Results", "/Users/katie/Desktop/CoopFiles/DyeImages/PlotValues" + i + "b.csv");

selectWindow("dye" + i + ".png (red)");
close();
selectWindow("Plot of dye" + i + ".png (red)");
close();
selectWindow("PlotValues" + i + "b.csv");
close("PlotValues" + i + "b.csv");
}

