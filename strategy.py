#!/usr/bin/env python3
# 6h_RSI_Streak_Contrarian_1dTrend_Volume
# Hypothesis: Contrarian mean reversion on 6h using RSI streak (consecutive up/down days) filtered by daily trend and volume.
# In strong trends (above/below daily EMA34), extended RSI streaks (>4) signal exhaustion and potential reversal.
# Volume confirmation ensures the move has participation. Works in both bull/bear by following higher timeframe trend.
# Target: 15-30 trades/year (60-120 over 4 years) with strict entry conditions to minimize fee drag.

name = "6h_RSI_Streak_Contrarian_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter and volume context
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # RSI(14) calculation on 6h closes
    def rsi(close_prices, period=14):
        delta = np.diff(close_prices)
        up = np.where(delta > 0, delta, 0)
        down = np.where(delta < 0, -delta, 0)
        roll_up = np.zeros_like(close_prices)
        roll_down = np.zeros_like(close_prices)
        for i in range(period, len(close_prices)):
            roll_up[i] = np.mean(up[i-period:i])
            roll_down[i] = np.mean(down[i-period:i])
        rs = np.where(roll_down != 0, roll_up / roll_down, 0)
        rsi_vals = np.full_like(close_prices, 50.0)
        rsi_vals[period:] = 100 - (100 / (1 + rs[period:]))
        return rsi_vals
    
    rsi_vals = rsi(close, 14)
    
    # RSI streak: consecutive days above/below 50
    rsi_streak = np.zeros_like(close)
    current_streak = 0
    for i in range(1, len(close)):
        if rsi_vals[i] > 50 and rsi_vals[i-1] > 50:
            current_streak += 1
        elif rsi_vals[i] < 50 and rsi_vals[i-1] < 50:
            current_streak -= 1
        else:
            current_streak = 1 if rsi_vals[i] > 50 else -1 if rsi_vals[i] < 50 else 0
        rsi_streak[i] = current_streak
    
    # Daily volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume_1d, 20)
    
    # Align daily indicators to 6h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI streak negative (downtrend exhaustion) in uptrend, with volume
            if rsi_streak[i] <= -4 and close[i] > ema_34_1d_aligned[i] and volume[i] > 1.5 * vol_ma_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI streak positive (uptrend exhaustion) in downtrend, with volume
            elif rsi_streak[i] >= 4 and close[i] < ema_34_1d_aligned[i] and volume[i] > 1.5 * vol_ma_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI returns to neutral or trend breaks
            if rsi_streak[i] >= 0 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI returns to neutral or trend breaks
            if rsi_streak[i] <= 0 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals