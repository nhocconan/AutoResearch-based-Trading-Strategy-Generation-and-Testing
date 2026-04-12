#!/usr/bin/env python3
# 6h_1d_mrsi_reversion
# Hypothesis: Modified RSI (MRSI) on daily timeframe identifies overbought/oversold conditions.
# In 6h timeframe, we mean-revert when MRSI is extreme and price touches Bollinger Bands.
# Works in bull/bear because it fades extremes rather than following trend.
# Uses volume confirmation to avoid false signals in low volatility.
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.

name = "6h_1d_mrsi_reversion"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for MRSI and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Modified RSI (MRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    # RSI(3)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/3, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/3, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi3 = 100 - (100 / (1 + rs))
    
    # RSI Streak (2): count consecutive up/down days
    streak = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        if close_1d[i] > close_1d[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close_1d[i] < close_1d[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    # RSI of streak (clamped to -20,20 for RSI calculation)
    streak_clipped = np.clip(streak, -20, 20)
    # Calculate RSI on streak (using same 3-period)
    delta_streak = np.diff(streak_clipped, prepend=streak_clipped[0])
    gain_streak = np.where(delta_streak > 0, delta_streak, 0)
    loss_streak = np.where(delta_streak < 0, -delta_streak, 0)
    avg_gain_streak = pd.Series(gain_streak).ewm(alpha=1/3, adjust=False).mean().values
    avg_loss_streak = pd.Series(loss_streak).ewm(alpha=1/3, adjust=False).mean().values
    rs_streak = avg_gain_streak / (avg_loss_streak + 1e-10)
    rsi_streak = 100 - (100 / (1 + rs_streak))
    
    # Percent Rank (100): where today's close ranks in last 100 days
    def rolling_percent_rank(arr, window):
        from scipy.stats import rankdata
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            window_data = arr[i-window+1:i+1]
            # Rank of last element in window (0-100)
            rank = (rankdata(window_data, method='average')[-1] - 1) / (window-1) * 100
            result[i] = rank
        return result
    percent_rank = rolling_percent_rank(close_1d, 100)
    
    # MRSI = average of three components
    mrsi = (rsi3 + rsi_streak + percent_rank) / 3
    
    # Bollinger Bands (20, 2)
    sma20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    
    # Align indicators to 6h timeframe
    mrsi_aligned = align_htf_to_ltf(prices, df_1d, mrsi)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(mrsi_aligned[i]) or np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: MRSI oversold (<30) and price touches lower BB
        if (mrsi_aligned[i] < 30 and close[i] <= lower_aligned[i] and 
            vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: MRSI overbought (>70) and price touches upper BB
        elif (mrsi_aligned[i] > 70 and close[i] >= upper_aligned[i] and 
              vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: MRSI returns to neutral zone (40-60) or opposite touch
        elif position == 1 and (mrsi_aligned[i] > 40 or close[i] >= upper_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (mrsi_aligned[i] < 60 or close[i] <= lower_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals