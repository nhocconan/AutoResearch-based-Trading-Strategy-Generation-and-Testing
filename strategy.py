#!/usr/bin/env python3
"""
12h_1w_ema_trend_follow_v1
Trend following on 12h timeframe using 21-period EMA on weekly close.
Long when price > EMA, short when price < EMA.
Entries only when price touches EMA (pullback) with volume confirmation.
Designed for low trade frequency (~15-25/year) to minimize fee drag.
Works in both bull and bear markets by following the trend defined by weekly EMA.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_ema_trend_follow_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 21-period EMA on weekly close
    close_1w = df_1w['close'].values
    ema_21 = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align EMA to 12h timeframe (waits for weekly bar to close)
    ema_21_aligned = align_htf_to_ltf(prices, df_1w, ema_21)
    
    # Volume filter: 24-period average on 12h timeframe
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if EMA or volume data is invalid
        if np.isnan(ema_21_aligned[i]) or np.isnan(vol_avg_24[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 24-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_24[i]
        
        # Price touching EMA (within 0.5% for long, 0.5% below for short)
        ema_touch_long = low[i] <= ema_21_aligned[i] * 1.005 and high[i] >= ema_21_aligned[i] * 0.995
        ema_touch_short = high[i] >= ema_21_aligned[i] * 0.995 and low[i] <= ema_21_aligned[i] * 1.005
        
        # Trend direction based on EMA slope (using previous value to avoid look-ahead)
        if i > 30:
            ema_slope = ema_21_aligned[i] - ema_21_aligned[i-1]
            trend_up = ema_slope > 0
            trend_down = ema_slope < 0
        else:
            trend_up = True  # Default to allow initial entry
            trend_down = False
        
        # Entry conditions
        # Long: Price touches EMA from below AND uptrend AND volume confirmation
        if ema_touch_long and trend_up and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Price touches EMA from above AND downtrend AND volume confirmation
        elif ema_touch_short and trend_down and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Price crosses EMA in opposite direction
        elif position == 1 and high[i] < ema_21_aligned[i] * 0.995:  # Cross below EMA
            position = 0
            signals[i] = 0.0
        elif position == -1 and low[i] > ema_21_aligned[i] * 1.005:  # Cross above EMA
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals