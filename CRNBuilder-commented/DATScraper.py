import numpy as np
import os


def scrapeDAT(case_name):
    print("Starting dat scraping...")

    # Load raw data
    print(os.path.join(f"Opening {os.getcwd()}", f"data", f"{case_name}", f"{case_name}.dat..."))
    with open(os.path.join(f"{os.getcwd()}", f"data", f"{case_name}", f"{case_name}.dat")) as dataFile:
        dataLines = dataFile.readlines()

    # Get variable names
    dataNames = (dataLines.pop(0)).split()

    # Initialize dataSets dictionary
    dataSets = {name: np.empty(len(dataLines), dtype="float") for name in dataNames}

    # Check if data is already present, in case load it
    if os.path.isdir(os.path.join(f"{os.getcwd()}", f"data", f"{case_name}", f"cache", f"")):
        print("Cached data found. Loading it...")
        for dataSetName in dataSets:
            dataSets[dataSetName] = np.load(os.path.join(f"{os.getcwd()}", f"data", f"{case_name}", f"cache", f"{dataSetName}.npy"))
    else:
        print("Cached data not found. Scraping raw data...")
        # Interpret raw data
        for i in range(len(dataLines)):
            dataLine = dataLines[i]
            data = dataLine.split()
            for j in range(len(data)):
                dataSets[dataNames[j]][i] = float(data[j])
            print(f"\r{i / len(dataLines) * 100:.2f}%", end="")
        # Save data for future use
        print("\rSaving data...")
        os.mkdir(os.path.join(f"{os.getcwd()}", f"data", f"{case_name}", f"cache", f""))
        for dataSetName in dataSets:
            np.save(os.path.join(f"{os.getcwd()}", f"data", f"{case_name}", f"cache", f"{dataSetName}.npy"), dataSets[dataSetName])

    print("Loaded data.")

    return dataSets, dataNames


if __name__ == '__main__':
    caseName = "HM1Flame-Cuoci"
    dataSets, dataNames = scrapeDAT(case_name=caseName)
