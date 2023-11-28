This is composed with two parts, FD and classfier

For FD, run FD_Mutual_Info.py, notice, for every dataset there is a config file, right now I have only finished config for nypd. The attribute that required for the FD search is listed in the read config function. After the run is compelted, the output will be in FD_result and FD_Graphs, where FD list is the list that contains all relations in this dataset

For classfiers, run k_prototype.py to save the clustered data into Dataset/k_proto_clustered folder
Then use Classifier.py to save classfiers into Save_Classifier folder