#!/usr/bin/env python3
# 12h_1w1d_Trend_With_Volume_Confirmation
# Hypothesis: Use weekly EMA50 as primary trend filter and daily EMA20 as entry filter on 12h timeframe.
# Long when weekly trend is up and price is above daily EMA20 with volume confirmation.
# Short when weekly trend is down and price is below daily EMA20 with volume confirmation.
# Designed for low trade frequency (~15-25 trades/year) to minimize fee drag and work in both bull and bear markets.

name = "12h_1w1d_Trend_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for primary trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 trend
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1w_up = close_1w > ema50_1w
    trend_1w_down = close_1w < ema50_1w
    
    # Daily data for entry filter and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily EMA20 for entry
    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Daily volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly trend to 12h
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    
    # Align daily EMA20 and volume average to 12h
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i]) or
            np.isnan(ema20_1d_aligned[i]) or np.isnan(vol_avg_1d_aligned[i]) or
            np.isnan(volume[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: weekly uptrend, price above daily EMA20, volume above average
            if (close[i] > ema20_1d_aligned[i] and
                trend_1w_up_aligned[i] > 0.5 and
                volume[i] > vol_avg_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend, price below daily EMA20, volume above average
            elif (close[i] < ema20_1d_aligned[i] and
                  trend_1w_down_aligned[i] > 0.5 and
                  volume[i] > vol_avg_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below daily EMA20 or weekly trend turns down
            if (close[i] < ema20_1d_aligned[i] or
                trend_1w_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above daily EMA20 or weekly trend turns up
            if (close[i] > ema20_1d_aligned[i] or
                trend_1w_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals