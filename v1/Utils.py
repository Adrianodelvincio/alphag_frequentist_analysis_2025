import pandas as pd

def ReadTheCsv(filename, window, zmin, zmax, CutsType):
    df = pd.read_csv(filename, delimiter=",") # read the dataset
    df = df.rename(columns={"Detector Time (detectors internal clock)": "Detector Time", "OfficialTime (run time)": "run time"})
    mask = (df[CutsType] == 1) & (df["Z"] > zmin) & ( df["Z"] < zmax)
    df_cleaned =  df.dropna()#[mask] # remove the Missing values for the dataset
    return df[mask]

def ReadTheSpill(filename, window):
    # select the window events
    print(filename)
    occurrencies = 0
    with open(filename.replace(".vertex.csv", "spilllog.log.txt"), "r") as file:
        for line in file:
            if(window in line):
                print(line)
                occurrencies += 1
                if(occurrencies > 1): print(bcolors.WARNING + f"ATTENTION, TWO ENTRIES IN SPILLOG {filename}" + bcolors.RESET)
    
    stop  = []
    start = []
    counter = 1
    if(occurrencies < 1):
        print(f"window {window} NOT FOUND")
    with open(filename.replace(".vertex.csv", "spilllog.log.txt"), "r") as file:
        for line in file:
            if (window in line):
                counter += 1
                if(counter >0):
                    parts = line.strip().split("=")
                    starter = parts[0].replace('[', '').replace(']','').split("-")[0]
                    stopper = parts[0].replace('[', '').replace(']','').split("-")[1]
                    start.append(float(starter))
                    stop.append(float(stopper))
                    #print(f"   {parts[0]},{parts[1]}...")
 
    return start, stop

def append_to_dict(dictionary, key, item):
    if key in dictionary.keys():
        dictionary[key].append(item)
    else:
        dictionary.update({key : [item]})