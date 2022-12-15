import numpy as np
import pandas as pd
from alphaminer.backtesting.metrics.cum_ret_groups import index_group_return, cum_ret_groups_corr
from alphaminer.backtesting.metrics.common import r2, tail_ratio, gain_to_pain_ratio, common_sense_ratio, beta,\
    ann_cum_ret


def benchmark_metrics(config_name, analysis_df, ic_df, rankic_df, pred_label_df, report_normal_df):
    # compute benchmark metrics
    mean, std = analysis_df.loc[("excess_return_without_cost", "mean")]['risk'], \
                analysis_df.loc[("excess_return_without_cost", "std")]['risk']
    ann_ret, max_dr = analysis_df.loc[("excess_return_without_cost", "annualized_return")]['risk'], \
                      analysis_df.loc[("excess_return_without_cost", "max_drawdown")]['risk']
    mean_wc, std_wc = analysis_df.loc[("excess_return_with_cost", "mean")]['risk'], \
                      analysis_df.loc[("excess_return_with_cost", "std")]['risk']
    ann_ret_wc, max_dr_wc = analysis_df.loc[("excess_return_with_cost", "annualized_return")]['risk'], \
                            analysis_df.loc[("excess_return_with_cost", "max_drawdown")]['risk']

    # for groups correlation
    _, indexes = index_group_return(pred_label_df)
    cum_ret_corr = cum_ret_groups_corr(indexes)

    r = report_normal_df["return"] - report_normal_df["bench"]
    r_wc = report_normal_df["return"] - report_normal_df["bench"] - report_normal_df["cost"]

    # for Sortino ratio
    r_neg = r[(r < 0)]
    r_neg_std = r_neg.std()
    r_wc_neg = r_wc[r_wc < 0]
    r_wc_neg_std = r_wc_neg.std()

    cum_ex_ret = np.cumsum(r)
    cum_ex_ret_wc = np.cumsum(r_wc)
    cum_ret = np.cumsum(report_normal_df["return"])
    cum_ret_wc = np.cumsum(report_normal_df["return"] - report_normal_df["cost"])

    results = {
        'config': config_name,
        'return_mean': mean,
        'return_std': std,
        'annualized_return': ann_ret,
        'information_ratio': analysis_df.loc[("excess_return_without_cost", "information_ratio")]['risk'],
        'max_drawdown': max_dr,
        'return_mean_wc': mean_wc,
        'return_std_wc': std_wc,
        'annualized_return_wc': ann_ret_wc,
        'information_ratio_wc': max_dr_wc,
        'max_drawdown_wc': analysis_df.loc[("excess_return_with_cost", "max_drawdown")]['risk'],
        'ic_mean': ic_df.mean(),
        'rank_ic_mean': rankic_df.mean(),
        'sharpe_ratio': mean / std,
        'y_sharpe_ratio': mean / std * np.sqrt(252),
        'sharpe_ratio_wc': mean_wc / std_wc,
        'y_sharpe_ratio_wc': mean_wc / std_wc * np.sqrt(252),
        'cum_ret_groups_corr': cum_ret_corr,
        'calmar_ratio': ann_ret / max_dr,
        'calmar_ratio_wc': ann_ret_wc / max_dr_wc,
        'sortino_ratio': mean / r_neg_std,
        'sortino_wc': mean_wc / r_wc_neg_std,
        'y_sortino_ratio': mean / r_neg_std * np.sqrt(252),
        'y_sortino_wc': mean_wc / r_wc_neg_std * np.sqrt(252),
        'cumret_&_time_r2_score': r2(np.array(list(range(len(r)))).reshape(-1, 1),
                                     cum_ret),  # r2 score of cum_return and time
        'tail_ratio': tail_ratio(r),
        'gain_to_pain_ratio': gain_to_pain_ratio(r),
        'common_sense_ratio': common_sense_ratio(r),
        'beta': beta(r, report_normal_df["bench"]),
        'cum_ret': cum_ret[-1],
        'cum_ret_wc': cum_ret_wc[-1],
        'cum_ex_ret': cum_ex_ret[-1],
        'cum_ex_ret_wc': cum_ex_ret_wc[-1],
        'y_cum_ex_ret': ann_cum_ret(cum_ex_ret[-1], r.index.year[-1] - r.index.year[0] + 1),
        'y_cum_ex_ret_wc': ann_cum_ret(cum_ex_ret_wc[-1], r.index.year[-1] - r.index.year[0] + 1),
    }
    return results


def detailed_report(report_normal_df, positions, pred_df):
    df_stats = report_normal_df.copy().reset_index()
    df_trades = pred_df.copy()
    dates = np.unique(df_trades.index.get_level_values(level=0))
    df_trades = df_trades.reset_index().dropna()
    amount, price, weight, count = [], [], [], []
    for date in dates:
        p = positions[pd.Timestamp(date)].position
        p_stocks = p.keys() - {"cash", "now_account_value", "cash_delay"}
        stocks = df_trades[df_trades['datetime'] == date]['instrument']
        for stock in stocks:
            if stock in p_stocks:
                amount.append(p[stock]['amount'])  # the amount of the security
                price.append(p[stock]['price'])  # the close price of security in the last trading day
                weight.append(p[stock]['weight'])  # the security weight of total position value
                count.append(p[stock]['count_day'])  # how many days the security has been hold
            else:
                amount.append(0.)
                price.append(0.)
                weight.append(0.)
                count.append(0.)
    df_trades['amount'] = amount
    df_trades['price'] = price
    df_trades['weight'] = weight
    df_trades['count_day'] = count

    return df_stats, df_trades
