import csv
# knapsack problem with more constraints
def optimalLineupCaptainMode(playerCosts, remainingSalary, fantasyPointsValue, n, playerIds, currentRoster):
    if n == 0 or remainingSalary == 0 or len(currentRoster) == 5:
        return 0, []

    if playerCosts[n-1] > remainingSalary:
        return optimalLineupCaptainMode(playerCosts, remainingSalary, fantasyPointsValue, n-1, playerIds, currentRoster)

    exclude_points, exclude_roster = optimalLineupCaptainMode(
        playerCosts, remainingSalary, fantasyPointsValue, n-1, playerIds, currentRoster
    )

    include_points, include_roster = optimalLineupCaptainMode(
        playerCosts, remainingSalary - playerCosts[n-1], fantasyPointsValue, n-1, playerIds, currentRoster + [playerIds[n-1]])

    if include_points + fantasyPointsValue[n-1] > exclude_points:
        include_roster.append(playerIds[n-1])
        return include_points + fantasyPointsValue[n-1], include_roster
    else:
        return exclude_points, exclude_roster



def removeElementFromArrAtIndex(arr, index):
    if 0 <= index < len(arr):
        return arr[:index] + arr[index + 1:]
    else:
        return arr  # Index out of range, return the original array

def main():
    # load from csv file
    # Example usage:
    # fppg =              [49.83,    
    #                      34,     
    #                      52.88,     
    #                      39.35,     
    #                      23.63,     
    #                      26.5,   
    #                      21.19,     
    #                      35.88,  
    #                      23.5, 32.65, 19.56 , 23.9, 19.69, 13.3,16.88, 12.88, 21.83,21.94]
    # playerCost =        [9200,   
    #                      7200,    
    #                      9600,    
    #                      8200,    
    #                      6800,    
    #                      7800,    
    #                      6000,    
    #                      7000,    
    #                      5800, 6600, 5600, 5000, 5200, 4200, 4600, 3800,4000,5400]
    # playerIds =         ['Tyrese Haliburton',   
    #                      'Myles Turner',    
    #                      'Donovan Mitchell',    
    #                      'Evan Mobley',    
    #                      'Buddy Hield',    
    #                      'Darius Garland',    
    #                      'Bruce Brown',    
    #                      'Caris LeVert',    
    #                      'T.J. McConnell', 'Max Strus', 'Bennedict Mathurin', 'Isaac Okoro', 'Andrew Nembhard', 'Georges Niang', 'Obi Toppin', 'Dean Wade','Jalen Smith',
    #                      'Aaron Nesmith']
    fppg =              [49.83,    
                         34,     
                         52.88,     
                         39.35,     
                         23.63,     
                         26.5,   
                         21.19,     
                         20,  
                         23.5, 20, 19.56 , 15, 19.69, 13.3,16.88, 12.88, 21.83,21.94]
    playerCost =        [9200,   
                         7200,    
                         9600,    
                         8200,    
                         6800,    
                         7800,    
                         6000,    
                         7000,    
                         5800, 6600, 5600, 5000, 5200, 4200, 4600, 3800,4000,5400]
    playerIds =         ['Tyrese Haliburton',   
                         'Myles Turner',    
                         'Donovan Mitchell',    
                         'Evan Mobley',    
                         'Buddy Hield',    
                         'Darius Garland',    
                         'Bruce Brown',    
                         'Caris LeVert',    
                         'T.J. McConnell', 'Max Strus', 'Bennedict Mathurin', 'Isaac Okoro', 'Andrew Nembhard', 'Georges Niang', 'Obi Toppin', 'Dean Wade','Jalen Smith',
                         'Aaron Nesmith']
    # playerIds = [
    #     "Tyrese Haliburton",
    #     "Myles Turner",
    #     "Buddy Hield",
    #     "Bruce Brown",
    #     "T.J. McConnell",
    #     "Bennedict Mathurin",
    #     "Aaron Nesmith",
    #     "Jalen Smith",
    #     "Donovan Mitchell",
    #     "Evan Mobley",
    #     "Darius Garland",
    #     "Caris LeVert",
    #     "Max Strus",
    #     "Jarrett Allen",
    #     "Isaac Okoro"
    #     ]
    # playerCost = [9200,
    #                 7200,
    #                 6800,
    #                 6000,
    #                 5800,
    #                 5600,
    #                 5400,
    #                 4000,
    #                 9600,
    #                 8200,
    #                 7800,
    #                 7000,
    #                 6600,
    #                 6400,
    #                 5000
    #                 ]
    # fppg = [43.5,
    #         46.75,
    #         26.75,
    #         31.25,
    #         0,
    #         18.25,
    #         25.25,
    #         23.25,
    #         60.25,
    #         48.5,
    #         27,
    #         27,
    #         15.25,
    #         24.25,
    #         13.25
    # ]
    
    for i, player in enumerate(playerIds):
        currentRoster = []
        maxSalary =     50000
        captain = player
        cptPointsAdded = 1.5*fppg[i]
        newSalary = maxSalary - 1.5*playerCost[i]
        playerIdNew = removeElementFromArrAtIndex(playerIds, i)
        playerCostNew = removeElementFromArrAtIndex(playerCost, i)
        fppgNew = removeElementFromArrAtIndex(fppg, i)
        n = len(fppgNew)
        max_points, selected_players = optimalLineupCaptainMode(playerCostNew, newSalary, fppgNew, n, playerIdNew, currentRoster)
        print(f"Captain: {captain}, Lineup \t{sorted(selected_players)}, projected pts: {max_points + cptPointsAdded}")


if __name__=="__main__":
    main()