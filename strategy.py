#!/usr/bin/env python3
"""
12h_RCI_Pullback_With_Daily_Filter
Hypothesis: Enter on Relative Strength Index (RSI) pullbacks in the direction of the daily trend.
Long when RSI < 30 and daily EMA20 up; short when RSI > 70 and daily EMA20 down.
Uses 12h chart for entry timing and 1d trend filter to avoid counter-trend trades.
Designed for low trade frequency (target 20-50/year) with position size 0.25.
Works in bull/bear: daily trend filter ensures alignment with higher timeframe momentum.
"""

name = "12h_RCI_Pullback_With_Daily_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA20 for trend filter
    close_daily = df_daily['close'].values
    ema20_daily = np.full_like(close_daily, np.nan)
    if len(close_daily) >= 20:
        multiplier = 2.0 / (20 + 1)
        ema20_daily[19] = np.mean(close_daily[:20])
        for i in range(20, len(close_daily)):
            ema20_daily[i] = multiplier * close_daily[i] + (1 - multiplier) * ema20_daily[i-1]
    ema20_daily_aligned = align_htf_to_ltf(prices, df_daily, ema20_daily)
    
    # Calculate 12h RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(1, n):
        if i < 14:
            avg_gain[i] = np.mean(gain[:i+1]) if i > 0 else 0
            avg_loss[i] = np.mean(loss[:i+1]) if i > 0 else 0
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema20_daily_aligned[i]) or 
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI oversold (<30) + daily uptrend
            if rsi[i] < 30 and close[i] > ema20_daily_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) + daily downtrend
            elif rsi[i] > 70 and close[i] < ema20_daily_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI overbought (>70) OR daily trend turns down
            if rsi[i] > 70 or close[i] < ema20_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI oversold (<30) OR daily trend turns up
            if rsi[i] < 30 or close[i] > ema20_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals