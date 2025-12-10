## 获取模型预测值和目标值的步骤

1. 如果你想保存模型的预测值和目标值，在运行基准测试时应将`--save-true-pred`设置为`True`。
例如：
```shell
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ILI.csv" --strategy-args '{"horizon":60}' --model-name "time_series_library.DLinear" --model-hyper-params '{"batch_size": 64, "d_ff": 512, "d_model": 256, "lr": 0.01, "horizon": 60, "seq_len": 104}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "your_result_path" --save-true-pred True
```

2. 基准测试运行完成后，预测值和目标值会保存在结果文件中。例如，如果你设置`--save-path "your_result_path"`，结果文件将位于`/result/your_result_path`。

3. 结果文件为`.tar.gz`格式，你需要解压该文件才能访问结果。

4. 预测值存储在`"inference_data"`列中，目标值存储在`"actual_data"`列中。不过，你可能会发现它们显示为乱码，因为它们是用base64编码的。你可以使用下面的函数来获取解码后的预测值和目标值。

```python
import base64
import pickle

import numpy as np
import pandas as pd


def decode_data(filepath: str) -> pd.DataFrame:
    """
    加载结果文件，解码经过base64编码的推理数据列和实际数据列。

    :param filepath: 结果数据的路径。
    :return: 解码后的数据。
    """
    data = pd.read_csv(filepath)
    for index, row in data.iterrows():
        decoded_inference_data = base64.b64decode(row["inference_data"])
        decoded_actual_data = base64.b64decode(row["actual_data"])
        data.at[index, "inference_data"] = pickle.loads(decoded_inference_data)
        data.at[index, "actual_data"] = pickle.loads(decoded_actual_data)
    return data


'''
如果你想将解码后的数据保存为CSV文件，请按照以下步骤操作。

your_result_path = r"your_result_path/your_result.csv"
decoded_result = decode_data(your_result_path)
pd.set_option('display.width', None)  # 避免数据中出现省略号。
np.set_printoptions(threshold=np.inf)  # 避免数据中出现省略号。
decoded_result.to_csv("decoded_result.csv", index=None)
'''
```