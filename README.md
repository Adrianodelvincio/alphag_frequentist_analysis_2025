chuckman simulation from https://alphacpc05.cern.ch/elog/ALPHA/36819 ==> vertices.xlsx


v1/ contains the script analysis and data for the alphag release experiment with laser cooled population.
To generate the plots and Scurve read the v1/README.txt file and follow the instruction inside (I post them also here below:)

# To run the code

# Export the run .csv file with the following commands: (example with run 13708, substitute that number with one your interest)

[on alphasuperdaq:/alphasoft/]

- source agconfig.sh
- root
- TAGPlot ag
- ag.AddTimeGate(13708, 0, 1000000000)
- ag.ExportCSV("Users/Adriano/ag13708")
- .q
- cd Users/Adriano/
- mkdir ag13708/
- mv ag13708* ag13708/

# Now export the data from alphasuperdaq to eos, if you are working on swan or in your local machine. The command to use is scp. Like:

- scp -r alpha@alphasuperdaq:alphasoft/Users/Adriano/ag13708/ ./

* bonus: in my local folder I have already all the data, so you can use the command

- scp -r alpha@alphasuperdaq:alphasoft/Users/Adriano/ag*/ ./

# now download the spilllog of the run from the elog and place it in the same folder with the data. Rename it changing the "R" prefix to "ag"

# open the Preliminary_Scurve_2025.ipynb and follow the instructions in the code comments and run the notebook

# This will generate the Scurve and a preliminary analysis of the data (No complex fit, just counts)


###!!!achtung!!!
You may want to check use python packeg installed options if you want to run the code on swan. Be sure to check this option when you load the session 
