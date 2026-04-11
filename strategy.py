#!/usr/bin/env python3
# 4h_1w_maxima_minima_v1
# Strategy: 4h maxima/minima breakout with weekly trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Weekly price extremes act as strong support/resistance. In bull markets, price breaks above weekly high with volume, indicating continuation. In bear markets, price breaks below weekly low with volume, indicating continuation. Weekly trend filter ensures alignment with higher timeframe momentum. Low trade frequency (~20-40/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1w_maxima_minima_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend and level reference
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly high and low (for support/resistance levels)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below weekly EMA20
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Entry logic: weekly high/low breakout + volume + trend alignment
        if close[i] > weekly_high_aligned[i] and vol_confirm[i] and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif close[i] < weekly_low_aligned[i] and vol_confirm[i] and downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: price returns to weekly EMA20 (trend reversal signal)
        elif position == 1 and close[i] < ema_20_1w_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > ema_20_1w_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals