@echo off
chcp 65001 >nul
title SmartGuard-ReID 依赖一键安装脚本
echo ==============================================
echo          SmartGuard-ReID 环境部署脚本
echo ==============================================
echo.

:: 【关键】自动定位脚本所在目录，永远不会错
cd /d "%~dp0"
echo ✅ 已自动定位到项目目录：%cd%

:: 激活 Anaconda reid 环境
call conda activate reid
if %errorlevel% neq 0 (
    echo ❌ 错误：未找到 reid 环境，请用 Anaconda Prompt 运行此脚本！
    pause
    exit /b 1
)
echo ✅ 已成功激活 reid 环境

echo.
echo 🚀 正在安装依赖...
pip install -r requirements.txt

echo.
echo ==============================================
echo          依赖安装完成！
echo ==============================================
pause