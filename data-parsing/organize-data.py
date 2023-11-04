# 1. need salaries for players on draftkings
# 2. then grab average stats up to the date of the game
# 3. store in csv (easy), eventually throw in mysql
# 4. 
import requests

def get_player_stats(player_name):
    base_url = "https://www.balldontlie.io/api/v1/"
    endpoint = "players"
    params = {
        "search": player_name
    }

    response = requests.get(f"{base_url}{endpoint}", params=params)
    data = response.json()

    if data["meta"]["total_count"] == 0:
        print("Player not found.")
        return

    player_id = data["data"][0]["id"]
    endpoint = f"season_averages?player_ids[]={player_id}"

    response = requests.get(f"{base_url}{endpoint}")
    data = response.json()

    if len(data["data"]) == 0:
        print("No season average stats found for the player.")
        return

    season_stats = data["data"][0]
    return season_stats




def main():
    # Example usage
    # make list of all players that we want to train the network one
    # we require salaries for each player during each game
    player_name = "LeBron James"
    player_stats = get_player_stats(player_name)

    if player_stats:
        for k in player_stats.keys():
            print(k, player_stats[k])

if __name__=="__main__":
    main()