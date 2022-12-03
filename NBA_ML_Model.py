#!/usr/bin/env python3
import pandas as pd
from sklearn import tree
from sklearn.svm import SVC
from sklearn.tree import (
    export_text,
)  # you can use this to display the tree in text formats
from sklearn.metrics import accuracy_score

# from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support

START_YEAR, END_YEAR = 1973, 2004

# parameters for decisions tree will be criterion of gini and entropy
parameters_dt = {"criterion": ["gini", "entropy"]}
# parameters for knn will be n_neighbors from 3 to 10 and different distance metrics, p value 1 to 4
parameters_knn = {
    "n_neighbors": range(3, 11),
    "metric": ["minkowski"],
    "p": range(1, 5),
}

parameters_linear_svc = {
    "C": [0.1],
    "kernel": ["linear"],
}

# parameters_poly_svc = {
#     "C": [0.1, 1],
#     "degree": [2, 3, 4],
#     "coef0": [ 0.001, 0.01, 0.1, 1, 5, 10],
#     "kernel": ["poly"],
# }

# parameters_rbf_svc = {
#     "C": [0.1, 1],
#     "gamma": [0.0001, 0.001, 0.002, 0.01, 0.02, 0.03, 0.1, 0.2, 1, 2, 3],
#     "kernel": ["rbf"],
# }

# parameters_sigmoid_svc = {
#     "C": [0.1, 1],
#     "gamma": [0.0001, 0.001, 0.002, 0.01, 0.02, 0.03, 0.1, 0.2, 1, 2, 3],
#     "coef0": [ 0.001, 0.01, 0.1, 1, 5, 10],
#     "kernel": ["sigmoid"],
# }


# map the years to all the players in that year
years_and_players = {}
years_and_players_cleaned = {}

# data in allstar txt file that is not needed
irrelevant_allstar_data = "conference,leag,gp,minutes,pts,dreb,oreb,reb,asts,stl,blk,turnover,pf,fga,fgm,fta,ftm,tpa,tpm".split(
    ","
)

# ilkid and year ?
aggregated_dimensions_player_data = (
    "team,leag,firstname,lastname,fga,fgm,fta,ftm,tpa,tpm".split(",")
)

aggregated_allstar_dimensions = "firstname,lastname".split(",")

# Sensitivity (TPR or recall) = True Positive / True Positive + False Negative this is all correct predictions over all the all stars
# this is better than accuracy because correct/all predictions will result in high accuracy given a bad model
def main():
    # DATA CLEANING:
    # ilkid,year,firstname,lastname,team,leag,gp,minutes,pts,oreb,dreb,reb,asts,stl,blk,turnover,pf (needed?),(fga (attempted),fgm (made)),(fta (free throw att),ftm),(tpa,tpm)
    # attempts and made we can make percentages to reduce our dimensions **

    print("\n===ORIGINAL REGULAR SEASON DATA===: \n\n")
    player_regular_season_data = pd.read_csv("./data/player_regular_season.txt")
    print(player_regular_season_data)

    print(
        "\n===DIMENSIONALITY REDUCTION ON REGULAR SEASON DATA and REMOVE DATA BEFORE 73===: \n\n"
    )
    player_regular_season_data["field_goal_percentage"] = (
        player_regular_season_data["fgm"] / player_regular_season_data["fga"]
    ) * 100
    player_regular_season_data["field_throw_percentage"] = (
        player_regular_season_data["ftm"] / player_regular_season_data["fta"]
    ) * 100
    player_regular_season_data["three_point_percentage"] = (
        player_regular_season_data["tpm"] / player_regular_season_data["tpa"]
    ) * 100
    # if field_goal_percentage is nan, then replace with 0
    player_regular_season_data["field_goal_percentage"] = player_regular_season_data["field_goal_percentage"].fillna(0)
    player_regular_season_data["field_throw_percentage"] = player_regular_season_data["field_throw_percentage"].fillna(0)
    player_regular_season_data["three_point_percentage"] = player_regular_season_data["three_point_percentage"].fillna(0)
    player_regular_season_data["id"] = (
        player_regular_season_data["ilkid"]
        + "-"
        + player_regular_season_data["year"].astype(str)
    )
    iterableRows = player_regular_season_data.iterrows()
    for index, player in iterableRows:
        if player["year"] < START_YEAR or player["year"] > END_YEAR:
            player_regular_season_data = player_regular_season_data.drop(index)
            # i = player_regular_season_data[((player_regular_season_data.ilkid == player["ilkid"])
            # & (player_regular_season_data.year == player["year"]))].index
    for dim in aggregated_dimensions_player_data:
        player_regular_season_data = player_regular_season_data.drop(dim, axis=1)
    print(player_regular_season_data)

    # ilkid,year,firstname,lastname (maybe combine ilkid concat with year for single identification player)
    print("\n===ALLSTAR SEASON DATA===: \n\n")
    player_allstar_data = pd.read_csv("./data/player_allstar.txt")
    print(player_allstar_data)
    print("\n===DIMENSIONALITY REDUCTION ON ALLSTAR DATA===: \n\n")
    for x in irrelevant_allstar_data:
        player_allstar_data = player_allstar_data.drop(x, axis=1)

    player_allstar_data["ilkid"] = player_allstar_data["ilkid"].str.upper()
    player_allstar_data["ilkid"] = player_allstar_data["ilkid"].str.strip()
    player_allstar_data["y"] = (
        player_allstar_data["ilkid"] + "-" + player_allstar_data["year"].astype(str)
    )
    for dim in aggregated_allstar_dimensions:
        player_allstar_data = player_allstar_data.drop(dim, axis=1)
    print(player_allstar_data)

    # combine ilkid+year, drop firstname and lastname, all other data but add 0 or 1 where 0=non-allstar that season and 1=allstar that season.
    # get a set to put all of the player_allstar_data y values into
    allstars = set()
    for index, player in player_allstar_data.iterrows():
        allstars.add(player["y"])
    # create a new column in player data filled with all zeros unless the set contains the id then put 1
    player_regular_season_data["isAllStar"] = 0
    for index, player in player_regular_season_data.iterrows():
        if str(player["id"]) in allstars:
            player_regular_season_data.loc[index, "isAllStar"] = 1
    # Weed out bench players and non-contributors:
    #    1> player must have at least 55 games played
    player_regular_season_data_cleaned = player_regular_season_data.copy()
    for index, player in player_regular_season_data_cleaned.iterrows():
        if int(player["gp"]) < 55:
            player_regular_season_data_cleaned = (
                player_regular_season_data_cleaned.drop(index)
            )
            # i = player_regular_season_data_cleaned[((player_regular_season_data_cleaned.ilkid == player["ilkid"])
            #                                         & (player_regular_season_data_cleaned.year == player["year"]))]

    print(
        f"Total players considered for training/testing: {len(player_regular_season_data)}"
    )
    print(
        f"Total players considered training/testing *cleaned*: {len(player_regular_season_data_cleaned)}"
    )

    for i in range(START_YEAR, END_YEAR + 1):
        years_and_players[str(i)] = set()
        years_and_players_cleaned[str(i)] = set()

    for _, player in player_regular_season_data.iterrows():
        years_and_players[str(player["year"])].add(str(player["ilkid"]))
    for _, player in player_regular_season_data_cleaned.iterrows():
        years_and_players_cleaned[str(player["year"])].add(str(player["ilkid"]))

    data = player_regular_season_data.copy()
    data = data.drop("ilkid", axis=1)
    data = data.drop("id", axis=1)
    data_55gp = player_regular_season_data_cleaned.copy()
    data_55gp = data_55gp.drop("ilkid", axis=1)
    data_55gp = data_55gp.drop("id", axis=1)

    print("\n\n===data used for models===\n")
    print(data)
    print("Data with at least 55 games played:\n")
    print(data_55gp)

    # CROSS-VALIDATION:
    #   SPLIT: LOOCV
    #     one season as TEST set and all other seasons as TRAINING set
    #        Measure sensitivity for each and average at end
    print("\n\n===LEAVE ONE OUT CROSS-VALIDATION===\n")
    for i in range(START_YEAR, END_YEAR + 1):
        print(f"current season as TEST set: {i}")
        # get all of the players with matching year to str(i) as test set
        test_set = data[data["year"].astype(int) == i]
        test_set_55gp = data_55gp[data_55gp["year"].astype(int) == i]
        # get all other players as training set
        training_set = data[data["year"].astype(int) != i]
        training_set_55gp = data_55gp[data_55gp["year"].astype(int) != i]
        # now drop all the years from all data since its not needed other than identifying the player
        training_set = training_set.drop("year", axis=1)
        test_set = test_set.drop("year", axis=1)
        training_set_55gp = training_set_55gp.drop("year", axis=1)
        test_set_55gp = test_set_55gp.drop("year", axis=1)
        # now split the training set into X and y
        X_train = training_set.drop("isAllStar", axis=1)
        y_train = training_set["isAllStar"]
        X_test = test_set.drop("isAllStar", axis=1)
        y_test = test_set["isAllStar"]
        # now split the training set into X and y
        X_train_55gp = training_set_55gp.drop("isAllStar", axis=1)
        y_train_55gp = training_set_55gp["isAllStar"]
        X_test_55gp = test_set_55gp.drop("isAllStar", axis=1)
        y_test_55gp = test_set_55gp["isAllStar"]
        # now run the model on the training set and test it on the test set
        # three models: ID3 tree, Support Vector Machine, and KNN
        # ID3 tree
        print("\n\n===ID3 Tree===:")
        id3 = DecisionTreeClassifier()
        print("Regular Season Data:")
        report_metrics(id3, parameters_dt, X_train, y_train, X_test, y_test)
        print("Regular Season Data with at least 55 games played:")
        id3 = DecisionTreeClassifier()
        report_metrics(
            id3, parameters_dt, X_train_55gp, y_train_55gp, X_test_55gp, y_test_55gp
        )
        # KNN
        print("\n===KNN===:")
        knn = KNeighborsClassifier()
        print("Regular Season Data:")
        report_metrics(knn, parameters_knn, X_train, y_train, X_test, y_test)
        print("Regular Season Data with at least 55 games played:")
        knn = KNeighborsClassifier()
        report_metrics(
            knn, parameters_knn, X_train_55gp, y_train_55gp, X_test_55gp, y_test_55gp
        )
        # Support Vector Machine
        print("\n===Support Vector Machine===:")
        print("Regular Season Data (linear):")
        svm = SVC()
        report_metrics(svm, parameters_linear_svc, X_train, y_train, X_test, y_test)
        print("Regular Season Data with at least 55 games played (linear):")
        svm = SVC()
        report_metrics(
            svm,
            parameters_linear_svc,
            X_train_55gp,
            y_train_55gp,
            X_test_55gp,
            y_test_55gp,
        )
        
        # print("Regular Season Data (polynomial):")
        # svm = SVC()
        # report_metrics(svm, parameters_poly_svc, X_train, y_train, X_test, y_test)
        # print("Regular Season Data (Radial Basic Function):")
        # svm = SVC()
        # report_metrics(svm, parameters_rbf_svc, X_train, y_train, X_test, y_test)
        # print("Regular Season Data (sigmoid):")
        # svm = SVC()
        # report_metrics(svm, parameters_sigmoid_svc, X_train, y_train, X_test, y_test)
        # print("Regular Season Data with at least 55 games played (polynomial):")
        # svm = SVC()
        # report_metrics(
        #     svm,
        #     parameters_poly_svc,
        #     X_train_55gp,
        #     y_train_55gp,
        #     X_test_55gp,
        #     y_test_55gp,
        # )
        # print(
        #     "Regular Season Data with at least 55 games played (Radial Basic Function):"
        # )
        # svm = SVC()
        # report_metrics(
        #     svm,
        #     parameters_rbf_svc,
        #     X_train_55gp,
        #     y_train_55gp,
        #     X_test_55gp,
        #     y_test_55gp,
        # )
        # print("Regular Season Data with at least 55 games played (sigmoid):")
        # svm = SVC()
        # report_metrics(
        #     svm,
        #     parameters_sigmoid_svc,
        #     X_train_55gp,
        #     y_train_55gp,
        #     X_test_55gp,
        #     y_test_55gp,
        # )
        print("=====================================================")

    #  SPLIT: 3-fold cross validation
    # use a 10 year seasons
    # 73-79, 80-86, 87-93
    print("===3-fold cross validation===")
    split_73_79 = set()
    split_80_86 = set()
    split_87_93 = set()
    split_94_04 = set()

    for yearStr, players in years_and_players.items():
        year = int(yearStr)
        for player in players:
            if year >= 1973 and year <= 1979:
                split_73_79.add(yearStr + player)
            if year >= 1980 and year <= 1986:
                split_80_86.add(yearStr + player)
            if year >= 1987 and year <= 1993:
                split_87_93.add(yearStr + player)
            if year >= 1994 and year <= 2004:
                split_94_04.add(yearStr + player)

    print(f"size for 73-79: {len(split_73_79)}")
    print(f"size for 80-86: {len(split_80_86)}")
    print(f"size for 87-93: {len(split_87_93)}")
    print(f"size for 94-04: {len(split_94_04)}")

    split_73_79C = set()
    split_80_86C = set()
    split_87_93C = set()
    split_94_04C = set()
    for yearStr, players in years_and_players_cleaned.items():
        year = int(yearStr)
        for player in players:
            if year >= 1973 and year <= 1979:
                split_73_79C.add(yearStr + player)
            if year >= 1980 and year <= 1986:
                split_80_86C.add(yearStr + player)
            if year >= 1987 and year <= 1993:
                split_87_93C.add(yearStr + player)
            if year >= 1994 and year <= 2004:
                split_94_04C.add(yearStr + player)
    print(f"size for 73-79 *cleaned: {len(split_73_79C)}")
    print(f"size for 80-86 *cleaned: {len(split_80_86C)}")
    print(f"size for 87-93 *cleaned: {len(split_87_93C)}")
    print(f"size for 94-04 *cleaned: {len(split_94_04C)}")


def report_metrics(model, parameters, trainDataX, trainDataY, testDataX, testDataY):
    try:
        model = GridSearchCV(model, parameters, cv=None)
        model.fit(trainDataX, trainDataY)
        predicted = model.predict(testDataX)
        print(f"==Accuracy Score: {accuracy_score(testDataY, predicted)}")
        print(
            f"==Precision Score: {precision_recall_fscore_support(testDataY, predicted, average='binary')[0]}"
        )
        print(
            f"==Recall Score: {precision_recall_fscore_support(testDataY, predicted, average='binary')[1]}"
        )
        print(
            f"==F-score: {precision_recall_fscore_support(testDataY, predicted, average='binary')[2]}"
        )
        print(f"==Best parameters: {model.best_params_}")
    except ValueError as e:
            print("*SKIPPING THIS MODEL* ValueError: ", e)


main()
