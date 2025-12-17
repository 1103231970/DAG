# -*- coding: utf-8 -*-

import numpy as np
from typing import List, Dict, Tuple, Any

__all__ = [
    "mae",
    "mse",
    "rmse",
    "mape",
    "smape",
    "mase",
    "wape",
    "msmape",
    "mae_norm",
    "mse_norm",
    "rmse_norm",
    "mape_norm",
    "smape_norm",
    "mase_norm",
    "wape_norm",
    "msmape_norm",
]


def _error(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """Simple error"""
    return actual - predicted


def _percentage_error(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """Percentage error"""
    return (actual - predicted) / actual


def mse(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """Mean Squared Error"""
    return np.mean(np.square(_error(actual, predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """Root Mean Squared Error"""
    return np.sqrt(mse(actual, predicted))


def mae(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """Mean Absolute Error"""

    return np.mean(np.abs(_error(actual, predicted)))


def mase(
    actual: np.ndarray,
    predicted: np.ndarray,
    hist_data: np.ndarray,
    seasonality: int = 2,
    **kwargs
):
    """
    Mean Absolute Scaled Error
    Baseline (benchmark) is computed with naive forecasting (shifted by @seasonality)
    """
    if seasonality == 2:
        return -1
    scale = len(predicted) / (len(hist_data) - seasonality)

    dif = 0
    for i in range((seasonality + 1), len(hist_data)):
        dif = dif + abs(hist_data[i] - hist_data[i - seasonality])

    scale = scale * dif

    return (sum(abs(actual - predicted)) / scale)[0]


def mape(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """
    Mean Absolute Percentage Error
    Properties:
        + Easy to interpret
        + Scale independent
        - Biased, not symmetric
        - Undefined when actual[t] == 0
    """
    return np.mean(np.abs(_percentage_error(actual, predicted))) * 100


def smape(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """
    Symmetric Mean Absolute Percentage Error
    """
    return (
        np.mean(
            2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)))
        )
        * 100
    )


def wape(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """Masked weighted absolute percentage error (WAPE)

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
    Returns:
        torch.Tensor: masked mean absolute error
    """
    loss = np.sum(np.abs(actual - predicted)) / np.sum(np.abs(actual)) * 100
    return loss


def msmape(actual: np.ndarray, predicted: np.ndarray, epsilon: float = 0.1, **kwargs):
    """
    Function to calculate series wise smape values

    Parameters
    forecasts - a matrix containing forecasts for a set of series
                no: of rows should be equal to number of series and no: of columns should be equal to the forecast horizon
    test_set - a matrix with the same dimensions as 'forecasts' containing the actual values corresponding with them
    """

    comparator = np.full_like(actual, 0.5 + epsilon)
    denom = np.maximum(comparator, np.abs(predicted) + np.abs(actual) + epsilon)
    msmape_per_series = np.mean(2 * np.abs(predicted - actual) / denom) * 100
    return msmape_per_series


def _error_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    """Simple error"""
    return scaler.transform(actual) - scaler.transform(predicted)


def _percentage_error_norm(
    actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs
):
    """Percentage error"""
    return (scaler.transform(actual) - scaler.transform(predicted)) / scaler.transform(
        actual
    )


def mse_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    """Mean Squared Error"""
    return np.mean(np.square(_error_norm(actual, predicted, scaler)))


def rmse_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    """Root Mean Squared Error"""
    return np.sqrt(mse_norm(actual, predicted, scaler))


def mae_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    """Mean Absolute Error"""

    return np.mean(np.abs(_error_norm(actual, predicted, scaler)))


def mase_norm(
    actual: np.ndarray,
    predicted: np.ndarray,
    scaler: object,
    hist_data: np.ndarray,
    seasonality: int = 2,
    **kwargs
):
    """
    Mean Absolute Scaled Error
    Baseline (benchmark) is computed with naive forecasting (shifted by @seasonality)
    """
    actual = scaler.transform(actual)
    predicted = scaler.transform(predicted)
    hist_data = scaler.transform(hist_data)
    if seasonality == 2:
        return -1
    scale = len(predicted) / (len(hist_data) - seasonality)

    dif = 0
    for i in range((seasonality + 1), len(hist_data)):
        dif = dif + abs(hist_data[i] - hist_data[i - seasonality])

    scale = scale * dif

    return (sum(abs(actual - predicted)) / scale)[0]


def mape_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    """
    Mean Absolute Percentage Error
    Properties:
        + Easy to interpret
        + Scale independent
        - Biased, not symmetric
        - Undefined when actual[t] == 0
    """
    return np.mean(np.abs(_percentage_error_norm(actual, predicted, scaler))) * 100


def smape_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    """
    Symmetric Mean Absolute Percentage Error
    """
    actual = scaler.transform(actual)
    predicted = scaler.transform(predicted)
    return (
        np.mean(
            2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)))
        )
        * 100
    )


def wape_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    """Masked weighted absolute percentage error (WAPE)

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
    Returns:
        torch.Tensor: masked mean absolute error
    """
    actual = scaler.transform(actual)
    predicted = scaler.transform(predicted)
    loss = np.sum(np.abs(actual - predicted)) / np.sum(np.abs(actual)) * 100
    return loss


def msmape_norm(
    actual: np.ndarray,
    predicted: np.ndarray,
    scaler: object,
    epsilon: float = 0.1,
    **kwargs
):
    """
    Function to calculate series wise smape values

    Parameters
    forecasts - a matrix containing forecasts for a set of series
                no: of rows should be equal to number of series and no: of columns should be equal to the forecast horizon
    test_set - a matrix with the same dimensions as 'forecasts' containing the actual values corresponding with them
    """
    actual = scaler.transform(actual)
    predicted = scaler.transform(predicted)
    comparator = np.full_like(actual, 0.5 + epsilon)
    denom = np.maximum(comparator, np.abs(predicted) + np.abs(actual) + epsilon)
    msmape_per_series = np.mean(2 * np.abs(predicted - actual) / denom) * 100
    return msmape_per_series


def corr(pred, true):
    """
        单变量使用
        趋势相关指标运算：越接近1，趋势越一致
    """
    u = ((pred - pred.mean()) * (true - true.mean())).sum()  # 全局均值
    d = np.sqrt(((pred - pred.mean()) ** 2).sum() * ((true - true.mean()) ** 2).sum())
    return u / (d + 1e-8) if d != 0 else 0


# 整合指标名称和值
def metric_mapping(metric_names, values):
    metric_values = values[:len(metric_names)]  # 指标数值列表（顺序对应）
    result_dict = dict(zip(metric_names, metric_values))  # 一一映射为字典
    return result_dict


class Top10Correction:
    def __init__(self, corr_func=corr):
        """
        筛选相关性最高的前10个窗口

        :param corr_func: 相关性计算函数，默认使用timekan的CORR，可替换为pathformer等其他实现
        """
        self.top_windows: List[Tuple[float, float, float, float, np.ndarray, np.ndarray]] = []
        self.max_keep = 10  # 保留相关性最高的10个窗口
        self.corr_func = corr_func  # 允许外部指定相关性函数（适配不同模型）

    def put_sort(self, target: np.ndarray, predict: np.ndarray, mae: float, mse: float):
        corr_value = corr(predict, target)

        # 2. 过滤低CORR样本（仅保留趋势一致性高的）
        if corr_value < 0.6:  # 可根据数据调整阈值
            return

        # 3. 结合CORR和MSE排序：先按CORR降序，再按MSE升序
        # 存储为 (-corr, mse, ...)，排序时按第一个元素升序（等价于CORR降序），再按第二个元素升序（MSE升序）
        self.top_windows.append((-corr_value, mse, mae, corr_value, target.copy(), predict.copy()))
        # 排序逻辑：优先CORR高（-corr小），再MSE小
        self.top_windows.sort(key=lambda x: (x[0], x[1]))
        if len(self.top_windows) > self.max_keep:
            self.top_windows = self.top_windows[:self.max_keep]

    def get_results(self) -> List[Dict[str, Any]]:
        """
        返回Top10高相关度窗口的结果

        :return: 包含每个窗口的相关性、MAE、MSE、真实值、预测值的列表
        """
        return [
            {
                "mse": item[1],  # MSE值
                "mae": item[2],  # MAE值
                "corr": item[3],  # 相关性值（越大越好）
                "target": item[4],  # 真实值序列
                "predict": item[5]  # 预测值序列
            }
            for item in self.top_windows
        ]
