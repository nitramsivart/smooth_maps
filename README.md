A method of plotting geographically distributed data on the US map using:
- Kernel density estimation to spread out data spatially
- Bayesian estimates of posteriors to correct noise in areas with very little data

Input (see data.input):
A tsv where each line contains four data points: longitude, latitude, number of 'yes' answers, total number of answers

Output (see out/):
A smoothed US map of the input data

Takes up to 3 command line parameters:
- data file name
- plot title
- output file (defaults to 'out/[plot title].png')

More parameters can be customized in the code.

Requires:
- python3
- numpy, scipy, matplotlib
- Basemap

To test, run:
python main.py
