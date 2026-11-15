@echo off
chcp 65001 >nul
title SmartGuard-ReID 一键演示
cd /d "%~dp0"
call conda activate reid
if %errorlevel% neq 0 (
    echo [错误] 未找到 reid 环境，请用 Anaconda Prompt 运行此脚本！
    pause
    exit /b 1
)
python retrieve.py
echo.
echo 结果图已保存到 outputs\ 文件夹，按任意键退出...
pause >nul
