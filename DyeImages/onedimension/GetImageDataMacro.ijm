for (i=6920; i <= 6920; i=i+1){
filename = "IMG_" + i + ".JPG";

open("/Users/katie/Desktop/CoopFiles/DyeImages/onedimension/Images/" + filename);
run("Split Channels");
selectWindow(filename + " (red)");
close();
selectWindow(filename + " (blue)");
close();
selectWindow(filename + " (green)");
//setTool("multipoint");

makeLine(486, 1422, 2154, 1350);

run("Plots...", "width=1000 height=340 font=14 draw_ticks list minimum=0 maximum=0 interpolate");
run("Plot Profile");
selectWindow("Plot Values");
saveAs("Results", "/Users/katie/Desktop/CoopFiles/DyeImages/onedimension/PlotValues" + i + ".csv");

selectWindow(filename);
close();
selectWindow("Plot of " + "IMG_" + i);
close();
selectWindow("PlotValues" + i + ".csv");
close("PlotValues" + i + ".csv");
}

