#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend direction with 1w EMA34 filter and volume confirmation.
# Enter long when KAMA(14,2,30) slope > 0, 1w EMA34 trending up, and volume > 1.5x 20-bar average.
# Enter short when KAMA slope < 0, 1w EMA34 trending down, and volume > 1.5x 20-bar average.
# Exit when KAMA slope reverses or price crosses 1w EMA34.
# Uses discrete position sizing (0.25) to minimize fee drag and manage drawdown.
# Target: 50-100 total trades over 4 years (12-25/year) to avoid excessive fee churn.
# KAMA adapts to market efficiency, reducing whipsaws in ranging markets.
# 1w EMA34 provides strong trend filter for multi-week alignment.
# Volume confirmation ensures breakouts have institutional participation.

name = "1d_KAMA_Trend_1wEMA34_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    er = np.zeros_like(change)
    for i in range(1, len(change)):
        if volatility[i] != 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 1.0
    # 10-period ER for fast, 30-period for slow
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34
    close_1w = df_1w['close'].values
    ema_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # KAMA trend: slope over 3 periods
        if i >= 3:
            kama_slope = (kama[i] - kama[i-3]) / 3
            kama_trend_up = kama_slope > 0
            kama_trend_down = kama_slope < 0
        else:
            kama_trend_up = False
            kama_trend_down = False
        
        # Exit conditions: KAMA slope reverses or price crosses 1w EMA34
        exit_long = kama_slope < 0 or close[i] < ema_34_aligned[i]
        exit_short = kama_slope > 0 or close[i] > ema_34_aligned[i]
        
        # Handle entries and exits
        if kama_trend_up and vol_confirm and position <= 0:
            signals[i] = 0.25
            position = 1
        elif kama_trend_down and vol_confirm and position >= 0:
            signals[i] = -0.25
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals