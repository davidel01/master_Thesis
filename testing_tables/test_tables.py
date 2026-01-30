table_A = [
    ["Pos", 1, 2, 3, 4],
    ["Club", "Korça", "Vlora", "Berati", "Gjirokastra"],
    ["P", 6, 6, 6, 6],
    ["W", 4, 3, 2, 0],
    ["D", 1, 2, 1, 2],
    ["L", 1, 1, 3, 4],
    ["A", 13, 17, 11, 17],
    ["Pts", 9, 8, 5, 2]
]

table_B = [
    ["Pos", 1, 2, 3, 4],
    ["Team", "Pennine Pumas", "Saxon Tigers", "Caledonian Cougars", "Celtic Panthers"],
    ["W", 3, 2, 1, 0],
    ["P", 3, 3, 3, 3],
    ["D", 0, 0, 0, 0],
    ["Pts", 6, 4, 2, 0],
    ["L", 0, 1, 2, 3],
    ["F", 7, 9, 4, 0],
    ["A", 0, 4, 7, 9],
]

table_stocks_a = [
    ["2024-03-01", "2024-03-04", "2024-03-06", "2024-03-07", "2024-03-08", "2024-03-11", "2024-03-13"],  # Date
    ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL", "AAPL", "AAPL"],                                            # Ticker
    [182.64, 181.95, 182.78, 185.33, 187.35, 189.20, 189.12],                                            # Open_Price
    [181.56, 184.37, 185.24, 187.92, 188.67, 187.11, 188.34],                                            # Close_Price
    ["-0.59%", "+1.55%", "+1.43%", "+1.45%", "+0.40%", "-0.83%", "-0.64%"],                              # Perc_Change
    [2856432000, 2900578000, 2914275000, 2956321000, 2968125000, 2943578000, 2962987000]                 # Market_Cap
]


table_stocks_b = [
    ["2024-03-01", "2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07", "2024-03-08", "2024-03-09", "2024-03-11", "2024-03-12"],  # Date
    [182.64, 181.95, 410.20, 182.78, 185.33, 187.35, 415.20, 189.20, 417.32],                                                       # Open_Price
    ["AAPL", "AAPL", "MSFT", "AAPL", "AAPL", "AAPL", "MSFT", "AAPL", "MSFT"],                                                       # Ticker
    [181.56, 184.37, 412.54, 185.24, 187.92, 188.67, 417.65, 187.11, 419.21],                                                       # Close_Price
    ["-0.59%", "+1.55%", "+1.59%", "+1.43%", "+1.45%", "+0.40%", "+0.59%", "-0.83%", "+0.89%"],                                     # Perc_Change
    [0.54, 0.53, 0.51, 0.43, 0.53, 0.53, 0.72, 0.53, 0.31],                                                                          # Dividend_Yield
    [2856432000, 2900578000, 3101561000, 2914275000, 2956321000, 2968125000, 3102500000, 2943578000, 3104680000]                    # Market_Cap
]

table_sport_a = [
    ["Arsenal", "Barcelona", "Bayern Munich", "Inter Milan", "Liverpool", "Manchester United", "Milan", "Real Madrid"],
    ["London", "Barcelona", "Munich", "Milan", "Liverpool", "Manchester", "Milan", "Madrid"],
    ["Emirates Stadium", "Camp Nou", "Allianz Arena", "San Siro", "Anfield", "Old Trafford", "San Siro", "Santiago Bernabéu"],
    [60704, 99354, 75000, 80018, 53394, 74310, 80018, 81044]
]

table_sport_b = [
    ["Real Madrid", "Milan", "Bayern Munich", "Liverpool", "Barcelona", "Ajax", "Inter Milan", "Manchester United", "Chelsea", "Juventus"],
    ["Madrid", "Milan", "Munich", "Liverpool", "Barcelona", "Amsterdam", "Milan", "Manchester", "London", "Turin"],
    ["Spain", "Italy", "Germany", "England", "Spain", "Netherlands", "Italy", "England", "England", "Italy"],
    ["Santiago Bernabéu", "San Siro", "Allianz Arena", "Anfield", "Camp Nou", "Johan Cruyff Arena", "Giuseppe Meazza", "Old Trafford", "Stamford Bridge", "Juventus Stadium"],
    [1902, 1899, 1900, 1892, 1899, 1900, 1908, 1878, 1905, 1897]
]

table_sport_a2 = [
    ["Arsenal", "Real Madrid", "Bayern Munich", "Inter Milan", "Liverpool", "Manchester United", "Milan"],
    ["London", "Madrid", "Munich", "Milan", "Liverpool", "Manchester", "Milan"],
    ["Emirates Stadium", "Camp Nou", "Allianz Arena", "San Siro", "Anfield", "Old Trafford", "San Siro"],
    [60704, 99354, 75000, 80018, 53394, 74310, 80018]
]

table_sport_b2 = [
    ["Real Madrid", "Milan", "Bayern Munich", "Liverpool", "Ajax", "Manchester United", "Inter Milan"],
    ["Madrid", "Milan", "Munich", "Liverpool", "Amsterdam", "Manchester", "Milan"],
    ["Spain", "Italy", "Germany", "England", "Netherlands", "England", "Italy"],
    ["Santiago Bernabéu", "San Siro", "Allianz Arena", "Anfield", "Johan Cruyff Arena", "Old Trafford", "Giuseppe Meazza"],
    [1902, 1899, 1900, 1892, 1900, 1878, 1908]
]

table_sport_a3 = [
    ["Real Madrid", "Real Madrid", "Bayern Munich", "Real Madrid"],
    ["Madrid", "Madrid", "Munich", "Madrid"],
    ["Camp Nou", "Camp Nou", "Allianz Arena", "Camp Nou"],
    [99354, 99354, 75000, 99354]
]

table_sport_b3 = [
    ["Real Madrid", "Milan", "Real Madrid", "Real Madrid"],
    ["Madrid", "Milan", "Madrid", "Madrid"],
    ["Spain", "Italy", "Spain", "Spain"],
    ["Camp Nou", "San Siro", "Santiago Bernabéu", "Santiago Bernabéu"],
    [1902, 1899, 1902, 1902]
]

table_sport_a4 = [
    ["Real Madrid", "Real Madrid", "Bayern Munich", "Real Madrid"],
    ["Madrid", "Madrid", "Munich", "Madrid"],
    ["Camp Nou", "Santiago Bernabéu", "Allianz Arena", "Santiago Bernabéu"],
    [99354, 99354, 75000, 99354]
]

table_sport_b4 = [
    ["Real Madrid", "Milan", "Real Madrid", "Real Madrid"],
    ["Madrid", "Milan", "Madrid", "Madrid"],
    ["Spain", "Italy", "Spain", "Spain"],
    ["Camp Nou", "San Siro", "Camp Nou", "Camp Nou"],
    [1902, 1899, 1902, 1902]
]