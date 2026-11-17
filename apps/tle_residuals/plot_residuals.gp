# Plot dense residual curves and residuals at the original TLE epochs.
#
# Interactive plots (default):
#   gnuplot -e "input='path/to/residuals.csv'" plot_residuals.gp
#
# PDF plots:
#   gnuplot -e "input='path/to/residuals.csv'; pdf=1" plot_residuals.gp

if (!exists("input")) {
    print "Set the input CSV with -e \"input='path/to/residuals.csv'\""
    exit
}

if (!exists("pdf")) pdf = 0

set datafile separator comma
set xdata time
set timefmt "%Y-%m-%dT%H:%M:%S"
set format x "%Y-%b-%d"
set xtics rotate by -45
set xlabel "Date (UTC)"
set grid
set key outside
set pointsize 0.7

# Remove the .csv suffix for the PDF filenames and find the dense table.
prefix = input[1:strlen(input)-4]
dense = sprintf("%s_dense.txt", prefix)

if (pdf) {
    set terminal pdfcairo enhanced color size 10in, 5.5in
    set output sprintf("%s_position.pdf", prefix)
} else {
    set terminal qt 0 enhanced
}

set ylabel "Position residual (m)"
plot dense using 1:3 with lines lc 1 title "R", \
     dense using 1:4 with lines lc 2 title "I", \
     dense using 1:5 with lines lc 3 title "C", \
     dense using 1:9 with lines lc 4 title "Norm", \
     input using 1:3 with points lc 1 notitle, \
     input using 1:4 with points lc 2 notitle, \
     input using 1:5 with points lc 3 notitle, \
     input using 1:9 with points lc 4 notitle

if (pdf) {
    set output sprintf("%s_velocity.pdf", prefix)
} else {
    set terminal qt 1 enhanced
}

set ylabel "Velocity residual (m/s)"
plot dense using 1:6 with lines lc 1 title "R", \
     dense using 1:7 with lines lc 2 title "I", \
     dense using 1:8 with lines lc 3 title "C", \
     dense using 1:10 with lines lc 4 title "Norm", \
     input using 1:6 with points lc 1 notitle, \
     input using 1:7 with points lc 2 notitle, \
     input using 1:8 with points lc 3 notitle, \
     input using 1:10 with points lc 4 notitle

if (!pdf) pause mouse close
