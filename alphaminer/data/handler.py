from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.handler import check_transform_proc
from typing import List, Optional, Union, Dict, Tuple, Any


_DEFAULT_LEARN_PROCESSORS = [
    #{"class": "DropnaLabel"},
    #{"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
]
_DEFAULT_INFER_PROCESSORS = [
    #{"class": "ProcessInf", "kwargs": {}},
    #{"class": "ZScoreNorm", "kwargs": {}},
    #{"class": "Fillna", "kwargs": {}},
]
basic_features = ["$close", "$factor", "$high", "$low", "$open", "$vwap"]


class AlphaMinerHandler(DataHandlerLP):
    def __init__(
        self,
        instruments: str = "csi500",
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        freq: str = "day",
        infer_processors=[],
        learn_processors=[],
        alphas: Optional[List[List[str]]] = None,
        fit_start_time: Optional[str] = None,
        fit_end_time: Optional[str] = None,
        filter_pipe=None,
        inst_processor=None,
        keep_basic_features: bool = True,
        **kwargs,
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)
        if alphas is not None:
            assert len(alphas) == 2
            assert len(alphas[0]) == len(alphas[1])
        features = alphas if alphas is not None else [basic_features, basic_features]
        if alphas and keep_basic_features:
            alphas[0] += basic_features
            alphas[1] += basic_features
        # TODO alphaminer doesn't require to have a label but Qlib does, so this field is here only for compatibility
        #  and can be ignored. Think how this can be fixed in a better way.
        labels = [["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL"]]

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": features,
                    "label": labels,
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processor": inst_processor,
            },
        }

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            learn_processors=learn_processors,
            infer_processors=infer_processors,
        )
