import os
import pandas as pd


INDEXES = ["deaths", "confirmed", "hospitalized", "intensive care", "intubated", "released"]


def read_inputs():

    text = input("Insert data to append as: [date: MM/DD/YY], [deaths], [confirmed], [hospitalized], [intensive care], "
                 "[intubated], [released]\n")

    data_list = text.split(" ")

    if len(data_list) != 7:
        raise ValueError("Missing input values.")

    new_data_series = pd.Series(index=INDEXES, data=data_list[1:])
    new_data_series.rename(data_list[0], inplace=True)
    new_data_series= new_data_series.astype('int')

    return new_data_series


def append_data(new_data_series):

    base_path = "../data"

    filepath_overall = os.path.join(base_path, "time_series_19-covid-all.csv")
    filepath_confirmed = os.path.join(base_path, "time_series_19-covid-Confirmed.csv")
    filepath_deaths = os.path.join(base_path, "time_series_19-covid-Deaths.csv")
    filepath_hospitalized = os.path.join(base_path, "time_series_19-covid-Hospitalized.csv")
    filepath_icu = os.path.join(base_path, "time_series_19-covid-ICU.csv")
    filepath_intubated = os.path.join(base_path, "time_series_19-covid-Intubated.csv")
    filepath_released = os.path.join(base_path, "time_series_19-covid-Released.csv")

    df_overall = pd.read_csv(filepath_overall)
    df_confirmed = pd.read_csv(filepath_confirmed)
    df_deaths = pd.read_csv(filepath_deaths)
    df_hospitalized = pd.read_csv(filepath_hospitalized)
    df_icu = pd.read_csv(filepath_icu)
    df_intubated = pd.read_csv(filepath_intubated)
    df_released = pd.read_csv(filepath_released)

    df_overall = df_overall.set_index("Type", drop=False)
    df_overall[new_data_series.name] = new_data_series
    df_confirmed[new_data_series.name] = new_data_series['confirmed']
    df_deaths[new_data_series.name] = new_data_series['deaths']
    df_hospitalized[new_data_series.name] = new_data_series['hospitalized']
    df_icu[new_data_series.name] = new_data_series['intensive care']
    df_intubated[new_data_series.name] = new_data_series['intubated']
    df_released[new_data_series.name] = new_data_series['released']

    df_overall.to_csv(filepath_overall, index=False)
    df_confirmed.to_csv(filepath_confirmed, index=False)
    df_deaths.to_csv(filepath_deaths, index=False)
    df_hospitalized.to_csv(filepath_hospitalized, index=False)
    df_icu.to_csv(filepath_icu, index=False)
    df_intubated.to_csv(filepath_intubated, index=False)
    df_released.to_csv(filepath_released, index=False)


if __name__ == '__main__':

    new_data_series = read_inputs()

    append_data(new_data_series)


