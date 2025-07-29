@echo off
setlocal enabledelayedexpansion

:: 設定參數
set TARGET_RATE=64000
set INPUT_DIR=.
set OUTPUT_DIR=
set TOTAL=0
set SUCCESS=0
set FAILED=0

:: 顯示說明
echo ====================================
echo FFmpeg 遞歸批次降採樣處理器
echo ====================================
echo.

:: 檢查FFmpeg是否安裝
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo 錯誤: 找不到 FFmpeg
    echo 請先安裝 FFmpeg: https://ffmpeg.org/download.html
    pause
    exit /b 1
)

:: 解析命令列參數
:parse_args
if "%~1"=="" goto start_processing
if "%~1"=="-i" (
    set INPUT_DIR=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="-o" (
    set OUTPUT_DIR=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="-r" (
    set TARGET_RATE=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="-h" goto show_help
if "%~1"=="--help" goto show_help
shift
goto parse_args

:start_processing
:: 如果沒指定輸出目錄，自動替換origin為64k
if "%OUTPUT_DIR%"=="" (
    set OUTPUT_DIR=%INPUT_DIR%
    set OUTPUT_DIR=!OUTPUT_DIR:\origin\=\64k\!
    if "!OUTPUT_DIR!"=="%INPUT_DIR%" (
        echo 警告: 輸入路徑不含 'origin'，使用預設輸出目錄
        set OUTPUT_DIR=%INPUT_DIR%_64k
    )
)

echo 輸入目錄: %INPUT_DIR%
echo 輸出目錄: %OUTPUT_DIR%
echo 目標採樣率: %TARGET_RATE% Hz
echo.

:: 掃描所有WAV檔案（遞歸）
echo 掃描檔案...
for /r "%INPUT_DIR%" %%f in (*.wav) do (
    set /a TOTAL+=1
)

if %TOTAL%==0 (
    echo 在 "%INPUT_DIR%" 及其子目錄找不到 WAV 檔案
    pause
    exit /b 1
)

echo 找到 %TOTAL% 個 WAV 檔案
echo.

:: 確認是否繼續
set /p CONFIRM=是否繼續處理? (Y/N):
if /i not "%CONFIRM%"=="Y" (
    echo 取消處理
    pause
    exit /b 0
)

:: 記錄開始時間
set START_TIME=%time%

:: 遞歸處理每個檔案
set CURRENT=0
for /r "%INPUT_DIR%" %%f in (*.wav) do (
    set /a CURRENT+=1
    set INPUT_FILE=%%f

    :: 計算相對路徑
    set REL_PATH=%%f
    set REL_PATH=!REL_PATH:%INPUT_DIR%\=!

    :: 生成輸出檔案路徑
    set OUTPUT_FILE=%OUTPUT_DIR%\!REL_PATH!

    :: 創建輸出目錄
    for %%d in ("!OUTPUT_FILE!") do (
        if not exist "%%~dpd" mkdir "%%~dpd" 2>nul
    )

    echo [!CURRENT!/%TOTAL%] 處理: !REL_PATH!

    :: 執行FFmpeg降採樣
    ffmpeg -hide_banner -loglevel error -i "!INPUT_FILE!" -ar %TARGET_RATE% -y "!OUTPUT_FILE!" 2>nul

    if !errorlevel!==0 (
        set /a SUCCESS+=1
        echo   ✓ 完成
    ) else (
        set /a FAILED+=1
        echo   ✗ 失敗: !INPUT_FILE!
    )
)

:: 計算處理時間
set END_TIME=%time%

:: 顯示結果
echo.
echo ====================================
echo 處理完成
echo ====================================
echo 總檔案: %TOTAL%
echo 成功: %SUCCESS%
echo 失敗: %FAILED%
echo 開始時間: %START_TIME%
echo 結束時間: %END_TIME%
echo.

if %FAILED% gtr 0 (
    echo 警告: %FAILED% 個檔案處理失敗
)

echo 輸出檔案位於: %OUTPUT_DIR%
pause
exit /b 0

:show_help
echo 用法: %~nx0 [選項]
echo.
echo 選項:
echo   -i ^<目錄^>     輸入目錄 (預設: 當前目錄)
echo   -o ^<目錄^>     輸出目錄 (預設: 自動替換origin為64k)
echo   -r ^<採樣率^>   目標採樣率 (預設: 64000)
echo   -h, --help    顯示此說明
echo.
echo 範例:
echo   %~nx0 -i "F:\...\origin"
echo   # 會自動輸出到 "F:\...\64k" 並保持目錄結構
echo.
echo   %~nx0 -i "F:\...\origin" -o "F:\...\processed"
echo   # 手動指定輸出目錄
echo.
echo 注意:
echo   - 會遞歸處理所有子目錄的WAV檔案
echo   - 自動保持原有目錄結構
echo   - 需要先安裝 FFmpeg
pause
exit /b 0