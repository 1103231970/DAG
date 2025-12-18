import pandas as pd
"""
    重要：要将指定预测的列放在格式化后数据集的最后
"""



# 上传文件目录
UPLOAD_DIR = 'upload/'
UPLOAD_FILENAME = 'test_dataset.csv'
# 上传文件中的时间列
TIME_COL_INDEX = 'date'
# 下载文件目录
DOWNLOAD_DIR = 'forecasting/'
DOWNLOAD_FILENAME = 'test_dataset.csv'


# 格式化数据集函数
def convert_to_tfb_series(data):
    data = data.set_index(TIME_COL_INDEX)  # time
    melted_df = data.melt(value_name="data", var_name="cols", ignore_index=False)  # data and col
    return melted_df.reset_index()[['date', 'data', 'cols']]


# ============================ 对数据进行格式化 =============================
data_source = pd.read_csv(UPLOAD_DIR + UPLOAD_FILENAME)
# 关键步骤：解析原始时间格式，并转换为目标格式
# 原始格式："2020年09月15日08时00分00秒" → 对应格式符："%Y年%m月%d日%H时%M分%S秒"
data_source[TIME_COL_INDEX] = pd.to_datetime(
    data_source[TIME_COL_INDEX],
    format="%Y年%m月%d日%H时%M分%S秒"
).dt.strftime("%Y-%m-%d %H:%M:%S")  # 转换为目标格式

data_source = data_source.dropna(how="all") # 过滤所有列为空的数据行
data_result = convert_to_tfb_series(data_source)
data_result.to_csv(DOWNLOAD_DIR+DOWNLOAD_FILENAME, index=False)
