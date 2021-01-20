#!/usr/bin/env python

import os
import argparse
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import git

_datetime_fmt = "%Y-%m-%d"

parser = argparse.ArgumentParser(description='Generate plots from COVID-19 data. Two sources available: local and OpenZH.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--source', type=str, default="OpenZH", choices=['local', 'OpenZH'], help='Data source')
parser.add_argument('--start_date', '-s', type=lambda s: datetime.datetime.strptime(s, _datetime_fmt), dest='start_datetime', default=datetime.datetime(2020, 2, 25).strftime(_datetime_fmt),
                    help=f"Plots start datetime (format: Y-m-d)")
parser.add_argument('--end_date', '-e', type=lambda s: datetime.datetime.strptime(s, _datetime_fmt), dest='end_datetime', default=datetime.datetime.now().strftime(_datetime_fmt),
                    help="Plots end datetime (format: Y-m-d)")
args = parser.parse_args()


SOURCE = args.source  # 'local' or 'OpenZH'
START_DATE = args.start_datetime
END_DATE = args.end_datetime
PLOT_PATH = "images"
OPENZH_REPO_DIR = 'covid_19'

FIGSIZE = (20, 10)
CANTONS_LIST = [
    {'name': "Ticino", 'abb': 'TI'},
    # {'name': "Zurich", 'abb': 'ZH'}
    # {'name': "Zug", 'abb': 'ZG'},
]
ALIGN_ZERO = False
PER_POPULATION = False

POPULATION = dict(  # in millions
    Ticino=0.353709,
    Zurich=1.521000
)


AVG_ROLLING_WINDOW = 7  # [3, 7]


def openzh_data_pull():

    print("Pulling OpenZH data...")
    openzh_git = git.cmd.Git(OPENZH_REPO_DIR)
    pull_response = openzh_git.pull()
    print(pull_response)


def load_data_from_source():

    df_confirmed = None
    df_deaths = None
    df_hospitalized = None
    df_icu = None
    df_intubated = None
    df_released = None
    df_positivity_rate = None

    if SOURCE == 'local':
        base_path = "data"

        filepath_confirmed = os.path.join(base_path, "time_series_19-covid-Confirmed.csv")
        filepath_deaths = os.path.join(base_path, "time_series_19-covid-Deaths.csv")
        filepath_hospitalized = os.path.join(base_path, "time_series_19-covid-Hospitalized.csv")
        filepath_icu = os.path.join(base_path, "time_series_19-covid-ICU.csv")
        filepath_intubated = os.path.join(base_path, "time_series_19-covid-Intubated.csv")
        filepath_released = os.path.join(base_path, "time_series_19-covid-Released.csv")

        df_confirmed = pd.read_csv(filepath_confirmed)
        df_deaths = pd.read_csv(filepath_deaths)
        df_hospitalized = pd.read_csv(filepath_hospitalized)
        df_icu = pd.read_csv(filepath_icu)
        df_intubated = pd.read_csv(filepath_intubated)
        df_released = pd.read_csv(filepath_released)

    elif SOURCE == 'OpenZH':

        # udate openzh data
        openzh_data_pull()

        for canton in CANTONS_LIST:
            base_path = f"{OPENZH_REPO_DIR}/fallzahlen_kanton_total_csv/COVID19_Fallzahlen_Kanton_{canton.get('abb')}_total.csv"

            df_canton = pd.read_csv(base_path)
            df_canton.set_index('date', inplace=True, drop=True)
            df_canton.index =pd.to_datetime(df_canton.index)

            df_confirmed = pd.DataFrame(df_canton['ncumul_conf'].rename(canton.get('name'))) if df_confirmed is None \
                else df_confirmed.join(df_canton['ncumul_conf'].rename(canton.get('name')))

            df_deaths = pd.DataFrame(df_canton['ncumul_deceased'].rename(canton.get('name'))) if df_deaths is None \
                else df_deaths.join(df_canton['ncumul_deceased'].rename(canton.get('name')))

            df_hospitalized = pd.DataFrame(df_canton['ncumul_hosp'].rename(canton.get('name'))) if df_hospitalized is None else \
                df_hospitalized.join(df_canton['ncumul_hosp'].rename(canton.get('name')))

            df_icu = pd.DataFrame(df_canton['ncumul_ICU'].rename(canton.get('name'))) if df_icu is None else \
                df_icu.join(df_canton['ncumul_ICU'].rename(canton.get('name')))

            df_intubated = pd.DataFrame(df_canton['ncumul_vent'].rename(canton.get('name'))) if df_intubated is None else \
                df_intubated.join(df_canton['ncumul_vent'].rename(canton.get('name')))

            # fix: for Ticino intubated moved from ncumul_vent to ninst_ICU_intub after 2020-03-23,08:00
            # mask_ncumul_ICU_intub_nan = df_canton['ninst_ICU_intub'].notna()
            # df_intubated.loc[mask_ncumul_ICU_intub_nan, canton.get('name')] = \
            #     df_intubated[mask_ncumul_ICU_intub_nan][canton.get('name')].fillna(0) + \
            #     df_canton['ninst_ICU_intub'][mask_ncumul_ICU_intub_nan]

            df_released = pd.DataFrame(
                df_canton['ncumul_released'].rename(canton.get('name'))) if df_released is None else \
                df_released.join(df_canton['ncumul_released'].rename(canton.get('name')))

            # tests
            base_path = f"{OPENZH_REPO_DIR}/fallzahlen_tests/fallzahlen_kanton_{canton.get('abb')}_tests.csv"
            df_canton = pd.read_csv(base_path, index_col='start_date', usecols=['start_date', 'total_tests', 'positivity_rate'])
            # df_canton = pd.read_csv(base_path, index_col='start_date', usecols=['start_date', 'positivity_rate'])
            df_canton.index = pd.to_datetime(df_canton.index)

            df_positivity_rate = pd.DataFrame(df_canton.add_prefix(canton.get('name') + "_")) if df_positivity_rate is None else \
                df_positivity_rate.join(df_canton.add_prefix(canton.get('name' + "_")))

    df_confirmed = df_confirmed[START_DATE: END_DATE]
    df_deaths = df_deaths[START_DATE: END_DATE]
    df_hospitalized = df_hospitalized[START_DATE: END_DATE]
    df_icu = df_icu[START_DATE: END_DATE]
    df_intubated = df_intubated[START_DATE: END_DATE]
    df_released = df_released[START_DATE: END_DATE]
    df_positivity_rate = df_positivity_rate[START_DATE:END_DATE]

    return df_confirmed, df_deaths, df_hospitalized, df_icu, df_intubated, df_released, df_positivity_rate


def clean_and_fix_data(df, align_zero=False, per_population=False):

    if SOURCE == 'local':

        country_region_list = [canton['name'] for canton in CANTONS_LIST]

        subset = df[df['Province/State'].isin(country_region_list)].copy()
        subset = subset.set_index('Province/State')

        cols_to_drop = ['Country/Region', 'Lat', 'Long']
        subset = subset.drop(columns=cols_to_drop)
        subset = subset.T
        subset.index = pd.to_datetime(subset.index)

    elif SOURCE == 'OpenZH':
        subset = df

    # fix missing data
    # if subset.notna().sum().min() > 3:
    # if subset.notna().sum().min() / len(subset) > 0.33:  # apply interpolation if at least a 1/3 of dates have data
    #     # subset = subset.interpolate(method='spline', order=2)
    #     subset = subset.interpolate(method='linear')

    subset = subset.ffill()

    if align_zero and len(CANTONS_LIST) > 1:

        support_df = subset[subset > 0]
        support_df = support_df.dropna(axis=0, how='all')

        ref_valid_index = support_df.first_valid_index()

        columns_suffix = []
        for col in support_df.columns:

            n_shift = support_df[col].isna().sum()
            # if support_df[col].max() > 100:
            #     n_shift = (support_df[col] < 10).sum()
            # else:
            #     n_shift = support_df[col].isna().sum()

            if n_shift:
                support_df[col] = np.roll(support_df[col], len(support_df) - n_shift)
                support_df.rename(columns={col: f"{col} (shifted {n_shift} days)"}, inplace=True)

        # raise NotImplementedError("Feature to be implemented.")
        support_df.reset_index(drop=True, inplace=True)
        support_df.index.name = "Days from first case."
        subset = support_df

    elif per_population:
        # case per 100k inhabitants

        for col in subset:
            subset[col] /= POPULATION.get(col) / 1e5

    # if avg_rolling_window:
    #     if isinstance(avg_rolling_window, int):
    #         avg_subset = subset.rolling(window=avg_rolling_window).mean()
    #         subset = subset.join(avg_subset.add_suffix(suffix=f" ({avg_rolling_window}-days avg)"))
    #
    #     elif isinstance(avg_rolling_window, list):
    #         _ref_subset = subset.copy()
    #         for _avg_rolling_window in avg_rolling_window:
    #             avg_subset = _ref_subset.rolling(window=_avg_rolling_window).mean()
    #             subset = subset.join(avg_subset.add_suffix(suffix=f" ({_avg_rolling_window}-days avg)"))

    return subset


def apply_avg(subset):

    if AVG_ROLLING_WINDOW:
        if isinstance(AVG_ROLLING_WINDOW, int):
            avg_subset = subset.rolling(window=AVG_ROLLING_WINDOW).mean()
            subset = subset.join(avg_subset.add_suffix(suffix=f" ({AVG_ROLLING_WINDOW}-days avg)"))

        elif isinstance(AVG_ROLLING_WINDOW, list):
            _ref_subset = subset.copy()
            for _avg_rolling_window in AVG_ROLLING_WINDOW:
                avg_subset = _ref_subset.rolling(window=_avg_rolling_window).mean()
                subset = subset.join(avg_subset.add_suffix(suffix=f" ({_avg_rolling_window}-days avg)"))

    return subset


def plot_multi(data, cols=None, spacing=.1, same_plot=False, **kwargs):
    """ref: https://stackoverflow.com/a/11643893/5490538"""

    if same_plot:
        # data.index = data.index.format()
        ax = data.plot(**kwargs)
        # ax.xaxis.set_major_locator(mdates.DayLocator())
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%b - %d'))
        lines, labels = ax.get_legend_handles_labels()
        ax.legend(lines, labels, loc=2)
        # ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
        # ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')

    else:
        plt.figure()
        # Get default color style from pandas - can be changed to any other color list
        if cols is None: cols = data.columns
        if len(cols) == 0: return
        # colors = getattr(getattr(plotting, '_matplotlib').style, '_get_standard_colors')(num_colors=len(cols))
        c_colors = plt.get_cmap("tab10")
        colors = c_colors.colors
        # First axis
        ax = data.loc[:, cols[0]].plot(label=cols[0], color=colors[0], **kwargs)
        ax.set_ylabel(ylabel=cols[0])
        lines, labels = ax.get_legend_handles_labels()

        for n in range(1, len(cols)):
            # Multiple y-axes
            ax_new = ax.twinx()
            ax_new.set_ylabel(ylabel=cols[n])

            ax_new.spines['right'].set_position(('axes', 1 + spacing * (n - 1)))
            data.loc[:, cols[n]].plot(ax=ax_new, label=cols[n], color=colors[n % len(colors)], **kwargs)
            # ax_new.set_ylabel(ylabel=cols[n])

            # Proper legend position
            line, label = ax_new.get_legend_handles_labels()
            lines += line
            labels += label

        ax.legend(lines, labels, loc=2)

    plt.gcf().autofmt_xdate()
    plt.grid(axis='y', color='0.95')

    if SOURCE == 'OpenZH':
        plt.savefig(os.path.join(PLOT_PATH, SOURCE + kwargs.get('title').replace(" ", "_") + ".png"))


if __name__ == '__main__':

    df_confirmed, df_deaths, df_hospitalized, df_icu, df_intubated, df_released, df_positivity_rate = load_data_from_source()

    df_confirmed = clean_and_fix_data(df_confirmed, align_zero=ALIGN_ZERO, per_population=PER_POPULATION)
    df_deaths = clean_and_fix_data(df_deaths, align_zero=ALIGN_ZERO, per_population=PER_POPULATION)
    df_hospitalized = clean_and_fix_data(df_hospitalized,align_zero=ALIGN_ZERO, per_population=PER_POPULATION)
    df_icu = clean_and_fix_data(df_icu, align_zero=ALIGN_ZERO, per_population=PER_POPULATION)
    df_intubated = clean_and_fix_data(df_intubated, align_zero=ALIGN_ZERO, per_population=PER_POPULATION)
    df_released = clean_and_fix_data(df_released, align_zero=ALIGN_ZERO, per_population=PER_POPULATION)

    df_hospitalized_plus_deaths = None
    df_hospitalized_plus_deaths_plus_released = None

    for each_canton in CANTONS_LIST:

        canton_name = each_canton['name']

        _df_hospitalized_plus_deaths = df_hospitalized[canton_name] + df_deaths[canton_name]
        _df_hospitalized_plus_deaths_plus_released = _df_hospitalized_plus_deaths + df_released[canton_name]

        if df_hospitalized_plus_deaths is None:
            df_hospitalized_plus_deaths = pd.DataFrame(_df_hospitalized_plus_deaths)

        else:
            df_hospitalized_plus_deaths[canton_name] = _df_hospitalized_plus_deaths

        if df_hospitalized_plus_deaths_plus_released is None:
            df_hospitalized_plus_deaths_plus_released = pd.DataFrame(_df_hospitalized_plus_deaths_plus_released)

        else:
            df_hospitalized_plus_deaths_plus_released[canton_name] = _df_hospitalized_plus_deaths_plus_released

    df_hospitalized = df_hospitalized.add_suffix(" - Hosp")

    df_hospitalized_plus_deaths = df_hospitalized_plus_deaths.add_suffix(" - Hosp+Deaths")
    df_hospitalized = df_hospitalized.join(df_hospitalized_plus_deaths)

    df_hospitalized_plus_deaths_plus_released = df_hospitalized_plus_deaths_plus_released.add_suffix(" - Hosp+Deaths+Released")
    df_hospitalized = df_hospitalized.join(df_hospitalized_plus_deaths_plus_released)

    # plot cumulative data
    plot_multi(apply_avg(df_confirmed), figsize=FIGSIZE, title="# of confirmed", same_plot=ALIGN_ZERO or PER_POPULATION or AVG_ROLLING_WINDOW, marker='o')
    plot_multi(apply_avg(df_deaths), figsize=FIGSIZE, title="# of deaths", same_plot=ALIGN_ZERO or PER_POPULATION or AVG_ROLLING_WINDOW, marker='o')
    plot_multi(apply_avg(df_hospitalized), figsize=FIGSIZE, title="# of hospitalized", same_plot=ALIGN_ZERO or PER_POPULATION or AVG_ROLLING_WINDOW, marker='o')
    plot_multi(apply_avg(df_icu), figsize=FIGSIZE, title="# of ICU", same_plot=ALIGN_ZERO or PER_POPULATION or AVG_ROLLING_WINDOW, marker='o')
    plot_multi(apply_avg(df_intubated), figsize=FIGSIZE, title="# of intubated", same_plot=ALIGN_ZERO or PER_POPULATION or AVG_ROLLING_WINDOW, marker='o')
    plot_multi(apply_avg(df_released), figsize=FIGSIZE, title="# of released", same_plot=ALIGN_ZERO or PER_POPULATION or AVG_ROLLING_WINDOW, marker='o')
    plot_multi(df_positivity_rate, figsize=FIGSIZE, title="positivity rate", same_plot=False, marker='o')

    # plot diff from previous day
    plot_multi(apply_avg(df_confirmed.diff()), figsize=FIGSIZE, title="Daily confirmed", same_plot=ALIGN_ZERO or PER_POPULATION or AVG_ROLLING_WINDOW, marker='o')
    plot_multi(apply_avg(df_deaths.diff()), figsize=FIGSIZE, title="Daily deaths", same_plot=ALIGN_ZERO or PER_POPULATION  or AVG_ROLLING_WINDOW, marker='o')
    plot_multi(apply_avg(df_hospitalized.diff()), figsize=FIGSIZE, title="Daily hospitalized", same_plot=ALIGN_ZERO or PER_POPULATION or AVG_ROLLING_WINDOW, marker='o')
    plot_multi(apply_avg(df_icu.diff()), figsize=FIGSIZE, title="Daily ICU", same_plot=ALIGN_ZERO or PER_POPULATION or AVG_ROLLING_WINDOW, marker='o')
    plot_multi(apply_avg(df_intubated.diff()), figsize=FIGSIZE, title="Daily intubated", same_plot=ALIGN_ZERO or PER_POPULATION or AVG_ROLLING_WINDOW, marker='o')
    plot_multi(apply_avg(df_released.diff()), figsize=FIGSIZE, title="Daily released", same_plot=ALIGN_ZERO or PER_POPULATION or AVG_ROLLING_WINDOW, marker='o')

    # case per day of week
    _df_confirmed = df_confirmed.diff()
    # assign monday date to dates within same week (monday to sunday)
    _df_confirmed['week'] = _df_confirmed.index - pd.to_timedelta(_df_confirmed.index.dayofweek, unit='d')
    _df_confirmed['day_of_week'] = _df_confirmed.index.day_name()
    df_confirmed_by_day_of_week = _df_confirmed.pivot(index='week', columns='day_of_week', values='Ticino')
    plot_multi(df_confirmed_by_day_of_week, figsize=FIGSIZE, title="Daily confirmed per Day Of Week", same_plot=ALIGN_ZERO or PER_POPULATION or AVG_ROLLING_WINDOW, marker='o')

    # # plot percentage changes day over day
    # plot_multi(df_confirmed.pct_change(),  figsize=FIGSIZE, title="Daily confirmed % growth change", same_plot=ALIGN_ZERO or PER_POPULATION)
    # plot_multi(df_deaths.pct_change(), figsize=FIGSIZE, title="Daily deaths % change", same_plot=ALIGN_ZERO or PER_POPULATION)
    # plot_multi(df_hospitalized.pct_change(), figsize=FIGSIZE, title="Daily recovered % change", same_plot=ALIGN_ZERO or PER_POPULATION)
    # plot_multi(df_icu.pct_change(), figsize=FIGSIZE, title="Daily ICU % change", same_plot=ALIGN_ZERO or PER_POPULATION)
    # plot_multi(df_intubated.pct_change(), figsize=FIGSIZE, title="Daily intubated % change", same_plot=ALIGN_ZERO or PER_POPULATION)

    # # plot percentage growth day over day
    # plot_multi(apply_avg(df_confirmed.cumsum().pct_change()), figsize=FIGSIZE, title="Confirmed % growth", same_plot=ALIGN_ZERO or PER_POPULATION or AVG_ROLLING_WINDOW)
    # plot_multi(apply_avg(df_deaths.cumsum().pct_change()), figsize=FIGSIZE, title="Deaths % growth", same_plot=ALIGN_ZERO or PER_POPULATION or AVG_ROLLING_WINDOW)
    # plot_multi(apply_avg(df_hospitalized.cumsum().pct_change()), figsize=FIGSIZE, title="Recovered % growth", same_plot=ALIGN_ZERO or PER_POPULATION or AVG_ROLLING_WINDOW)
    # plot_multi(apply_avg(df_icu.cumsum().pct_change()), figsize=FIGSIZE, title="ICU % growth", same_plot=ALIGN_ZERO or PER_POPULATION or AVG_ROLLING_WINDOW)
    # plot_multi(apply_avg(df_intubated.cumsum().pct_change()), figsize=FIGSIZE, title="Intubated % growth", same_plot=ALIGN_ZERO or PER_POPULATION or AVG_ROLLING_WINDOW)
    # plot_multi(apply_avg(df_released.cumsum().pct_change()), figsize=FIGSIZE, title="Released % growth", same_plot=ALIGN_ZERO or PER_POPULATION or AVG_ROLLING_WINDOW)

    # # plot percentage changes from cumsum
    # plot_multi(df_confirmed / df_confirmed.cumsum(),  figsize=FIGSIZE, title="Daily confirmed % over cumsum", same_plot=ALIGN_ZERO or PER_POPULATION)
    # plot_multi(df_deaths / df_deaths.cumsum(), figsize=FIGSIZE, title="Daily deaths % over cumsum", same_plot=ALIGN_ZERO or PER_POPULATION)
    # plot_multi(df_hospitalized / df_hospitalized.cumsum(), figsize=FIGSIZE, title="Daily recovered % over cumsum", same_plot=ALIGN_ZERO or PER_POPULATION)

    plt.show()
