#!/usr/bin/env python3
# 1d_1w_ema_cross_v1
# Hypothesis: Daily EMA crossover with weekly trend filter and volume confirmation.
# Long when daily EMA(9) crosses above EMA(21) AND weekly EMA(9) > EMA(21) (uptrend).
# Short when daily EMA(9) crosses below EMA(21) AND weekly EMA(9) < EMA(21) (downtrend).
# Exit when opposite crossover occurs.
# Uses volume > 1.5x 20-day average for confirmation.
# Works in both bull and bear markets as EMA adapts to price and weekly filter avoids counter-trend trades.
# Target: 50-100 total trades over 4 years (12-25/year) to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_ema_cross_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate daily EMA(9) and EMA(21)
    ema9 = np.full(n, np.nan)
    ema21 = np.full(n, np.nan)
    
    if n >= 9:
        ema9[8] = np.mean(close[:9])
        for i in range(9, n):
            ema9[i] = (close[i] * 2 / (9 + 1)) + (ema9[i-1] * (9 - 1) / (9 + 1))
    
    if n >= 21:
        ema21[20] = np.mean(close[:21])
        for i in range(21, n):
            ema21[i] = (close[i] * 2 / (21 + 1)) + (ema21[i-1] * (21 - 1) / (21 + 1))
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly EMA(9) and EMA(21)
    close_1w = df_1w['close'].values
    ema9_1w = np.full(len(close_1w), np.nan)
    ema21_1w = np.full(len(close_1w), np.nan)
    
    if len(close_1w) >= 9:
        ema9_1w[8] = np.mean(close_1w[:9])
        for i in range(9, len(close_1w)):
            ema9_1w[i] = (close_1w[i] * 2 / (9 + 1)) + (ema9_1w[i-1] * (9 - 1) / (9 + 1))
    
    if len(close_1w) >= 21:
        ema21_1w[20] = np.mean(close_1w[:21])
        for i in range(21, len(close_1w)):
            ema21_1w[i] = (close_1w[i] * 2 / (21 + 1)) + (ema21_1w[i-1] * (21 - 1) / (21 + 1))
    
    # Align weekly EMAs to daily timeframe
    ema9_1w_aligned = align_htf_to_ltf(prices, df_1w, ema9_1w)
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Volume confirmation - 20 period average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(ema9[i]) or np.isnan(ema21[i]) or np.isnan(ema9_1w_aligned[i]) or np.isnan(ema21_1w_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 1:  # Long position
            # Exit: daily EMA(9) crosses below EMA(21)
            if ema9[i] < ema21[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: daily EMA(9) crosses above EMA(21)
            if ema9[i] > ema21[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: daily EMA(9) crosses above EMA(21) AND weekly uptrend
            if ema9[i] > ema21[i] and ema9[i-1] <= ema21[i-1] and ema9_1w_aligned[i] > ema21_1w_aligned[i] and vol_ok:
                position = 1
                signals[i] = 0.25
            # Enter short: daily EMA(9) crosses below EMA(21) AND weekly downtrend
            elif ema9[i] < ema21[i] and ema9[i-1] >= ema21[i-1] and ema9_1w_aligned[i] < ema21_1w_aligned[i] and vol_ok:
                position = -1
                signals[i] = -0.25
    
    return signals