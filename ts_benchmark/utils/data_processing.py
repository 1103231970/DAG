# -*- coding: utf-8 -*-


from collections import OrderedDict
from typing import List, Optional, Tuple
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def _parse_target_channel(
    target_channel: Optional[List], num_columns: int
) -> List[int]:
    """
    Parses the target_channel configuration to determine target column indices.
    """
    if target_channel is None:
        return list(range(num_columns))  # Select all columns

    target_columns = []
    for item in target_channel:
        if isinstance(item, int):
            # Handle single integer index (supports negative indices)
            actual_index = item if item >= 0 else num_columns + item
            if 0 <= actual_index < num_columns:
                target_columns.append(actual_index)
            else:
                raise IndexError(
                    f"target_channel configuration error: Column index {item} is out of range (total columns: {num_columns})."
                )
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            # Handle slice represented as a list or tuple, e.g., [2, 4] or (2, 4) selects columns 2 and 3
            start, end = item
            start = start if start >= 0 else num_columns + start
            end = end if end >= 0 else num_columns + end

            if not (0 <= start < num_columns):
                raise IndexError(
                    f"target_channel configuration error: Slice start index {item[0]} is out of range (total columns: {num_columns})."
                )
            if not (0 <= end <= num_columns):
                raise IndexError(
                    f"target_channel configuration error: Slice end index {item[1]} is out of range (total columns: {num_columns})."
                )
            if start > end:
                raise ValueError(
                    f"target_channel configuration error: Slice start index {start} is greater than end index {end}."
                )

            # Add the range of indices to target_columns
            slice_indices = list(range(start, end))
            target_columns.extend(slice_indices)
        else:
            raise ValueError(
                f"target_channel configuration error: Invalid configuration item {item}."
            )

    # Remove duplicates while preserving order (using OrderedDict for compatibility with older Python versions)
    target_columns_unique = list(OrderedDict.fromkeys(target_columns))
    return target_columns_unique


def split_channel(
    df: pd.DataFrame, target_channel: Optional[List] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Splits a DataFrame into target and exogenous (exog) parts based on target_channel.

    :param df: Input DataFrame to split.
                              HUFL   HULL   MUFL   MULL   LUFL   LULL         OT
        date
        2016-07-01 00:00:00  5.827  2.009  1.599  0.462  4.203  1.340  30.531000
        2016-07-01 01:00:00  5.693  2.076  1.492  0.426  4.142  1.371  27.787001
        2016-07-01 02:00:00  5.157  1.741  1.279  0.355  3.777  1.218  27.787001
        2016-07-01 03:00:00  5.090  1.942  1.279  0.391  3.807  1.279  25.044001
        2016-07-01 04:00:00  5.358  1.942  1.492  0.462  3.868  1.279  21.948000
    :param target_channel: Rules for selecting target columns. Can include:

        - Integers (positive/negative) for single column indices.
        - Lists/tuples of two integers representing slices (e.g., `[2,4]` selects columns 2-3).
        - If `None`, all columns are treated as target columns (exog becomes None).

    :return: Tuple of (target_df, exog_df).
             exog_df is None if no exogenous columns exist.

             外生变量（exog_df）:
                                  HUFL   HULL    MUFL   MULL   LUFL   LULL
            date
            2016-07-01 00:00:00  5.827  2.009   1.599  0.462  4.203  1.340
            2016-07-01 01:00:00  5.693  2.076   1.492  0.426  4.142  1.371
            2016-07-01 02:00:00  5.157  1.741   1.279  0.355  3.777  1.218
            2016-07-01 03:00:00  5.090  1.942   1.279  0.391  3.807  1.279
            2016-07-01 04:00:00  5.358  1.942   1.492  0.462  3.868  1.279

            目标变量（target_df）:
                                        OT
            date
            2016-07-01 00:00:00  30.531000
            2016-07-01 01:00:00  27.787001
            2016-07-01 02:00:00  27.787001
            2016-07-01 03:00:00  25.044001
            2016-07-01 04:00:00  21.948000

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> df = pd.DataFrame(np.zeros((5, 6)))

        >>> # Case 1: Select columns 1 and 3
        >>> target, exog = split_channel(df, target_channel=[1, 3])
        >>> target.shape
        (5, 2)
        >>> exog.shape  # Exog contains remaining columns 0,2,4,5
        (5, 4)

        >>> # Case 2: Select columns 1-3 via slice
        >>> target, exog = split_channel(df, target_channel=[(1, 4)])
        >>> target.shape
        (5, 3)
        >>> exog.shape  # Exog contains columns 0,4,5
        (5, 3)

        >>> # Case 3: Select all columns when target_channel=None
        >>> target, exog = split_channel(df, target_channel=None)
        >>> target.shape  # All columns are target
        (5, 6)
        >>> print(exog)  # No exog columns
        None
    """
    logger.info("\n[All Columns Data] :%s\n%s", df.shape,df.head().to_string())
    num_columns = df.shape[1]  # Total number of columns in the DataFrame

    # Parse target_channel to get target column indices
    target_columns = _parse_target_channel(target_channel, num_columns)

    if target_channel is not None:
        # Determine exog columns by excluding target columns
        all_columns = set(range(num_columns))
        exog_columns = sorted(all_columns - set(target_columns))
    else:
        # If target_channel is None, exog_columns is empty
        exog_columns = []

    # Split the DataFrame into target and exog parts
    target_df = df.iloc[:, target_columns]
    exog_df = (
        df.iloc[:, exog_columns] if exog_columns else None
    )  # Directly return None if no exog columns

    # 关键：在占位符前加\n，并用to_string()格式化DataFrame
    logger.info("\n[Exo Columns Data] :%s \n%s", exog_df.shape,exog_df.head().to_string())
    logger.info("\n[Target Columns Data] :%s \n%s",target_df.shape,target_df.head().to_string())
    exit()
    return target_df, exog_df


def split_time(data: pd.DataFrame, index: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into two parts at the specified index.

    :param data: Time series data to be segmented.
    :param index: Split index position.
    :return: Split the first and second half of the data.
    """
    return data.iloc[:index, :], data.iloc[index:, :]
