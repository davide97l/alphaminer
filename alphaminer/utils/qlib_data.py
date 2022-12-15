import qlib
from qlib.data import D
import argparse


def qlib_data(
    start='2000-01-01',
    end='2022-07-01',
    features=None,
    data_source='~/.qlib/qlib_data/cn_data',
    qlib_normalization=False,
    save_path='qlib_data.csv'
):
    """
    Utility to retrieve data from Qlib as pd.Dataframe adapted to Alphaminer RL env format.
    Qlib data should be obtained from crowd source in order to preserve the quality of the data:
    https://github.com/microsoft/qlib/tree/main/scripts/data_collector/crowd_source.
    """
    qlib.init(provider_uri=data_source)
    if not features:
        features = ['$open', '$high', '$low', '$close', '$volume', '$factor']
    else:
        features = ['$' + f for f in features if '$' not in f]
    df = D.features(D.list_instruments(D.instruments()), features, start_time=start, end_time=end)
    if not qlib_normalization:
        df['$close'] = df['$close'] / df['$factor']
        df['$open'] = df['$open'] / df['$factor']
        df['$high'] = df['$high'] / df['$factor']
        df['$low'] = df['$low'] / df['$factor']
    df.reset_index(inplace=True)
    df.columns = ['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'factor']
    df["code"] = df["code"].apply(lambda x: x[:2].lower() + "." + x[2:])
    df["amount"] = 0.
    df["turn"] = 0.
    df["isST"] = 0.
    if save_path is not None:
        df.to_csv(save_path, index_label=None)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=str, default='2000-01-01')
    parser.add_argument('--end', type=str, default='2022-07-01')
    parser.add_argument('--data_source', type=str, default='~/.qlib/qlib_data/cn_data')
    parser.add_argument('--save_path', type=str, default='qlib_data.csv')
    parser.add_argument('--qlib_normalization', action='store_true')
    args = parser.parse_known_args()[0]
    qlib_data(
        start=args.start,
        end=args.end,
        data_source=args.data_source,
        save_path=args.save_path,
        qlib_normalization=args.qlib_normalization
    )
