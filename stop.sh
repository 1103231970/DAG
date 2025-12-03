#!/bin/bash

# 查找运行中的目标进程ID
PID=$(ps aux | grep 'DAG' | grep -v grep | awk '{print $2}')

if [ -n "$PID" ]; then
    echo "正在终止进程: $PID"
    kill $PID
    # 等待进程终止
    sleep 2
    # 检查进程是否已终止
    if ps -p $PID > /dev/null; then
        echo "进程未正常终止，尝试强制终止..."
        kill -9 $PID
    else
        echo "进程已成功终止"
    fi
else
    echo "未找到运行中的目标进程"
fi