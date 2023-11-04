import csv
import requests



def parseCsvIntoDict(filename, contestSet):
    # Open the CSV file for reading
    with open(filename, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        
        for row in csvreader:
            sport_name = row["Sport"]
            contest_key = row["Contest_Key"]
            entry_key = row["Entry_Key"]
            game_type = row["Game_Type"]
            if sport_name == "NBA" and game_type == "Showdown Captain Mode":
                # Check if the contest key is already in the dictionary
                if contest_key not in contestSet:
                    # If it's not, create a new entry in the dictionary
                    contestSet.add(contest_key)

def getJsonFromEndpoint(endpointUrl):
    data = {}
    # Send a GET request to the endpoint
    response = requests.get(endpointUrl)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
    else:
        # If the request was not successful, print an error message
        print(f"Request failed with status code: {response.status_code}")
    return data

def pullDkDataIntoPlayerDict(enpointUrl, fullPlayerDictOfAllPlayedGames):
    #testEndpoint = "https://api.draftkings.com/draftgroups/v1/draftgroups/94770/draftables"
    data = getJsonFromEndpoint(enpointUrl)
    draftables = data['draftables']
    playerSalaryDictSingleGame = {}
    for player in draftables:
        playerName = player['displayName']
        playerSalary = player['salary']
        playerTeam = player['teamAbbreviation']
        playerGameDate = player['competition']['startTime']
        isHome = False if (player['competition']['nameDisplay'][0]['value'] == playerTeam) else True
        oppTeamAbbreviation = player['competition']['nameDisplay'][0]["value"] if isHome else player['competition']['nameDisplay'][2]["value"] 
        playerObj = {
            'salary': playerSalary,
            'teamAbbreviation': playerTeam,
            'gameDate': playerGameDate,
            'isHome' : isHome,
            'oppTeamAbbreviation': oppTeamAbbreviation
        }
        if playerName in playerSalaryDictSingleGame:
            if playerSalaryDictSingleGame[playerName]['salary'] > playerSalary:
                playerSalaryDictSingleGame[playerName]['salary'] = playerSalary
        else:
            playerSalaryDictSingleGame[playerName] = playerObj

    for key in playerSalaryDictSingleGame.keys():
        if key in fullPlayerDictOfAllPlayedGames:
            fullPlayerDictOfAllPlayedGames[key].append(playerSalaryDictSingleGame[key])
        else:
            fullPlayerDictOfAllPlayedGames[key] = [playerSalaryDictSingleGame[key]]

def writeToCsv(filename, contestDict):
    # Open the CSV file for writing
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['contestId', 'draftGroupId']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header row
        writer.writeheader()

        # Write the data
        for key, value in contestDict.items():
            print(key)
            print(value)
            writer.writerow({'contestId': key, 'draftGroupId': value})

def parseContestIdToDict(filename, inputDict):
    # Open the CSV file for reading
    with open(filename, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        
        for row in csvreader:
            contestId = row["contestId"]
            draftId = row["draftGroupId"]
            inputDict[contestId] = draftId

def main():
    dkContestSet = set()
    fname = './draftkings-contest-entry-history.csv'
    contestIds = './contestIds.csv'
    inputDict = {}
    parseContestIdToDict(contestIds, inputDict)
    sorted_dict = {k: inputDict[k] for k in sorted(inputDict)}
    for key in sorted_dict.keys():
        print(f"contestid: {key}, {sorted_dict[key]}")
    # parseCsvIntoDict(fname, dkContestSet)
    #for each need to curl https://api.draftkings.com/contests/v1/contests/153510920?format=json | json_pp
    # then get the first element['contestDetail']['draftGroupId'] = 94770 example
    # then curl https://api.draftkings.com/draftgroups/v1/draftgroups/94770/draftables | json_pp
    # these are the player prices for that game

    # fullPlayerDictOfAllPlayedGames = {}
    # contestIdAndDraftGroupId = {}
    # for contestKey in dkContestSet: 
    #     print(f"contestKey number: {contestKey}")
    #     contestEndpoint = f"https://api.draftkings.com/contests/v1/contests/{contestKey}?format=json"
    #     contestJsonData = getJsonFromEndpoint(contestEndpoint)
    #     draftGroupId = contestJsonData['contestDetail']['draftGroupId']
    #     print(f"draftGroupId number: {draftGroupId}")
    #     draftTableEndpoint = f"https://api.draftkings.com/draftgroups/v1/draftgroups/{draftGroupId}/draftables"
    #     contestIdAndDraftGroupId[contestKey] = draftGroupId
    #     # pullDkDataIntoPlayerDict(draftTableEndpoint, fullPlayerDictOfAllPlayedGames)
    #     # print(f"pulled dk data for draftGroup: {draftGroupId}")
    # writeToCsv('./contestIds.csv', contestIdAndDraftGroupId)



    # for key in fullPlayerDictOfAllPlayedGames.keys():
    #     print(f"player: {key}, {fullPlayerDictOfAllPlayedGames[key]}")

if __name__=="__main__":
    main()