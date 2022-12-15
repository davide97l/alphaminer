from typing import List
import numpy as np
from alphaminer.factor.gtja_alpha191 import GTJA_191
from os import path
import pandas as pd


def check_alpha(alpha: pd.Series, min_count: int = 50, boundary: List[float] = [-1, 1]):
    """
    Default parameters should ensure the factor has no NaN value and fall in the range of -1 to 1.
    """
    print(alpha.describe().round(4))
    assert isinstance(alpha, pd.Series)
    assert alpha.dropna().shape[0] <= 50
    assert alpha.dropna().shape[0] >= min_count
    assert alpha.quantile(0.05) >= boundary[0]
    assert alpha.quantile(0.95) <= boundary[1]


def test_gtja_alpha191():
    # Init data
    test_dir = path.dirname(path.realpath(__file__))
    df = pd.read_csv(path.join(test_dir, "test_gtja_stocks.csv"))
    index_stocks = df["code"].unique().tolist()

    df["date"] = pd.to_datetime(df["date"])
    df["avg_price"] = (df["high"] + df["low"]) / 2
    df["amount"] = df["avg_price"] * df["volume"]
    df.set_index("date", inplace=True)
    df_index = pd.read_csv(path.join(test_dir, "test_gtja_index.csv"))
    df_index["date"] = pd.to_datetime(df_index["date"])
    df_index.set_index("date", inplace=True)

    # Start test
    gtja = GTJA_191("2022-08-01", security=index_stocks, price=df, benchmark_price=df_index)
    check_alpha(gtja.alpha_001())
    check_alpha(gtja.alpha_002(), min_count=48, boundary=[-100, 100])
    check_alpha(gtja.alpha_003(), boundary=[-300, 100])
    check_alpha(gtja.alpha_004(), boundary=[0, 10000])
    check_alpha(gtja.alpha_005())
    check_alpha(gtja.alpha_006())
    check_alpha(gtja.alpha_007(), boundary=[0, 2])
    check_alpha(gtja.alpha_008())
    check_alpha(gtja.alpha_009())
    check_alpha(gtja.alpha_010())
    check_alpha(gtja.alpha_011(), boundary=[-np.inf, np.inf])
    check_alpha(gtja.alpha_012())
    check_alpha(gtja.alpha_013(), boundary=[-20000, 0])
    check_alpha(gtja.alpha_014(), boundary=[-500, 30])
    check_alpha(gtja.alpha_015())
    check_alpha(gtja.alpha_016(), min_count=1, boundary=[-1, 0.1])
    check_alpha(gtja.alpha_017(), boundary=[0, np.inf])
    check_alpha(gtja.alpha_018(), boundary=[0, 2])
    check_alpha(gtja.alpha_019())
    check_alpha(gtja.alpha_020(), boundary=[-10, 20])
    check_alpha(gtja.alpha_021(), min_count=30, boundary=[-30, 160])
    check_alpha(gtja.alpha_022())
    check_alpha(gtja.alpha_023(), boundary=[30, 60])
    check_alpha(gtja.alpha_024(), boundary=[-300, 20])
    check_alpha(gtja.alpha_025())
    check_alpha(gtja.alpha_026(), boundary=[-20, 220])
    check_alpha(gtja.alpha_028(), boundary=[-10, 100])
    check_alpha(gtja.alpha_029(), boundary=[-np.inf, np.inf])
    check_alpha(gtja.alpha_031(), boundary=[-10, 10])
    check_alpha(gtja.alpha_032(), boundary=[-3, 0])
    check_alpha(gtja.alpha_033(), boundary=[-20, 350])
    check_alpha(gtja.alpha_034(), boundary=[0, 2])
    check_alpha(gtja.alpha_035())
    check_alpha(gtja.alpha_036(), min_count=10)
    check_alpha(gtja.alpha_037(), boundary=[-200, 400])
    check_alpha(gtja.alpha_038(), boundary=[-10, 2])
    check_alpha(gtja.alpha_039())
    check_alpha(gtja.alpha_040(), boundary=[0, 250])
    check_alpha(gtja.alpha_041())
    check_alpha(gtja.alpha_042())
    check_alpha(gtja.alpha_043(), boundary=[-np.inf, np.inf])
    check_alpha(gtja.alpha_044(), boundary=[0, 2])
    check_alpha(gtja.alpha_045())
    check_alpha(gtja.alpha_046(), boundary=[0, 2])
    check_alpha(gtja.alpha_047(), boundary=[0, 100])
    check_alpha(gtja.alpha_048())
    check_alpha(gtja.alpha_049())
    check_alpha(gtja.alpha_052(), boundary=[0, 7000])
    check_alpha(gtja.alpha_053(), boundary=[10, 70])
    check_alpha(gtja.alpha_054())
    check_alpha(gtja.alpha_056())
    check_alpha(gtja.alpha_057(), boundary=[0, 100])
    check_alpha(gtja.alpha_058(), boundary=[15, 60])
    check_alpha(gtja.alpha_059(), boundary=[-1300, 150])
    check_alpha(gtja.alpha_060(), boundary=[-np.inf, np.inf])
    check_alpha(gtja.alpha_061())
    check_alpha(gtja.alpha_062(), min_count=49)
    check_alpha(gtja.alpha_063(), boundary=[0, 100])
    check_alpha(gtja.alpha_064(), min_count=20)
    check_alpha(gtja.alpha_065(), boundary=[0, 2])
    check_alpha(gtja.alpha_066())
    check_alpha(gtja.alpha_067(), boundary=[35, 70])
    check_alpha(gtja.alpha_068())
    check_alpha(gtja.alpha_070(), boundary=[0, np.inf])
    check_alpha(gtja.alpha_071(), boundary=[-10, 10])
    check_alpha(gtja.alpha_072(), boundary=[30, 80])
    check_alpha(gtja.alpha_074(), min_count=35, boundary=[0, 2])
    check_alpha(gtja.alpha_075())
    check_alpha(gtja.alpha_076())
    check_alpha(gtja.alpha_077())
    check_alpha(gtja.alpha_078(), boundary=[-200, 210])
    check_alpha(gtja.alpha_079(), boundary=[20, 80])
    check_alpha(gtja.alpha_080(), boundary=[-50, 400])
    check_alpha(gtja.alpha_081(), boundary=[0, np.inf])
    check_alpha(gtja.alpha_082(), boundary=[40, 70])
    check_alpha(gtja.alpha_083())
    check_alpha(gtja.alpha_084(), boundary=[-np.inf, np.inf])
    check_alpha(gtja.alpha_085())
    check_alpha(gtja.alpha_086(), boundary=[-5, 15])
    check_alpha(gtja.alpha_087(), boundary=[0, 2])
    check_alpha(gtja.alpha_088(), boundary=[-20, 21])
    check_alpha(gtja.alpha_089(), boundary=[-200, 5])
    check_alpha(gtja.alpha_090(), min_count=18, boundary=[-1, 0])
    check_alpha(gtja.alpha_091())
    check_alpha(gtja.alpha_092(), min_count=49)
    check_alpha(gtja.alpha_093(), boundary=[0, 1400])
    check_alpha(gtja.alpha_094(), boundary=[-np.inf, np.inf])
    check_alpha(gtja.alpha_095(), boundary=[0, np.inf])
    check_alpha(gtja.alpha_096(), boundary=[0, 90])
    check_alpha(gtja.alpha_097(), boundary=[0, np.inf])
    check_alpha(gtja.alpha_098(), boundary=[-3000, 30])
    check_alpha(gtja.alpha_099())
    check_alpha(gtja.alpha_100(), boundary=[0, np.inf])
    # check_alpha(gtja.alpha_101()) # argument 0 of type numpy.float64 which has no callable rint method
    check_alpha(gtja.alpha_102(), boundary=[30, 100])
    check_alpha(gtja.alpha_103(), boundary=[0, 100])
    check_alpha(gtja.alpha_104(), boundary=[-1, 1.5])
    # check_alpha(gtja.alpha_105(), boundary=[-np.inf, np.inf], min_count=40) #nan occur
    check_alpha(gtja.alpha_106(), boundary=[-900, 50])
    check_alpha(gtja.alpha_107())
    check_alpha(gtja.alpha_108())
    check_alpha(gtja.alpha_109(), boundary=[0, 2])
    check_alpha(gtja.alpha_110(), boundary=[0, 200])
    check_alpha(gtja.alpha_111(), boundary=[-np.inf, np.inf])
    check_alpha(gtja.alpha_112(), boundary=[-100, 100])
    check_alpha(gtja.alpha_113(), min_count=40)
    check_alpha(gtja.alpha_114(), boundary=[-250, 50])
    check_alpha(gtja.alpha_115(), min_count=40)
    check_alpha(gtja.alpha_116(), min_count=40)
    check_alpha(gtja.alpha_117())
    check_alpha(gtja.alpha_118(), boundary=[-np.inf, np.inf])
    check_alpha(gtja.alpha_119(), boundary=[-1.0, 1.0], min_count=20)
    check_alpha(gtja.alpha_120(), boundary=[0, 50])
    check_alpha(gtja.alpha_122())
    check_alpha(gtja.alpha_123())
    check_alpha(gtja.alpha_124(), boundary=[-50, 50])
    check_alpha(gtja.alpha_125(), boundary=[0, 20])
    check_alpha(gtja.alpha_126(), boundary=[0, np.inf])
    check_alpha(gtja.alpha_129(), boundary=[0, 1000])
    check_alpha(gtja.alpha_130(), min_count=40, boundary=[0, 20])
    check_alpha(gtja.alpha_132(), boundary=[0, np.inf])
    check_alpha(gtja.alpha_133(), boundary=[-200, 200])
    check_alpha(gtja.alpha_134(), boundary=[-np.inf, np.inf])
    check_alpha(gtja.alpha_135(), boundary=[-1, 2])
    check_alpha(gtja.alpha_136())
    check_alpha(gtja.alpha_138(), min_count=3)
    check_alpha(gtja.alpha_139())
    check_alpha(gtja.alpha_140())
    check_alpha(gtja.alpha_141(), min_count=30)
    check_alpha(gtja.alpha_142())
    check_alpha(gtja.alpha_144(), min_count=40)
    check_alpha(gtja.alpha_145(), boundary=[-100, 50])
    check_alpha(gtja.alpha_148())
    check_alpha(gtja.alpha_150(), boundary=[0, np.inf])
    check_alpha(gtja.alpha_152())
    check_alpha(gtja.alpha_153(), boundary=[0, np.inf])
    # check_alpha(gtja.alpha_154()) #argument 0 of type numpy.float64 which has no callable rint method
    check_alpha(gtja.alpha_155(), boundary=[-np.inf, np.inf])
    check_alpha(gtja.alpha_156())
    check_alpha(gtja.alpha_157(), boundary=[0, 5])
    check_alpha(gtja.alpha_158())
    check_alpha(gtja.alpha_159(), boundary=[-np.inf, 0])
    check_alpha(gtja.alpha_160(), boundary=[0, 500])
    check_alpha(gtja.alpha_161(), boundary=[0, 300])
    check_alpha(gtja.alpha_162())
    check_alpha(gtja.alpha_163())
    check_alpha(gtja.alpha_164(), boundary=[0, 5000])
    check_alpha(gtja.alpha_167(), boundary=[0, 200])
    check_alpha(gtja.alpha_168(), boundary=[-5, 1])
    check_alpha(gtja.alpha_169(), boundary=[-50, 5])
    check_alpha(gtja.alpha_170())
    # check_alpha(gtja.alpha_171(), boundary=[-30, 0]), none occur
    check_alpha(gtja.alpha_172(), boundary=[5, 100])
    check_alpha(gtja.alpha_173(), boundary=[0, np.inf])
    check_alpha(gtja.alpha_174(), boundary=[0, 150])
    check_alpha(gtja.alpha_175(), boundary=[0, 250])
    check_alpha(gtja.alpha_176(), min_count=40)
    check_alpha(gtja.alpha_177(), boundary=[0, 100])
    check_alpha(gtja.alpha_178(), boundary=[-np.inf, np.inf])
    check_alpha(gtja.alpha_179())
    check_alpha(gtja.alpha_180(), boundary=[-np.inf, 1])
    # check_alpha(gtja.alpha_182())
    check_alpha(gtja.alpha_184(), boundary=[0, 2])
    check_alpha(gtja.alpha_185())
    check_alpha(gtja.alpha_186(), boundary=[0, 100])
    check_alpha(gtja.alpha_187(), boundary=[0, np.inf])
    check_alpha(gtja.alpha_188(), boundary=[-200, 200])
    check_alpha(gtja.alpha_189(), boundary=[-200, 200])
    check_alpha(gtja.alpha_191(), boundary=[-15, 30])
