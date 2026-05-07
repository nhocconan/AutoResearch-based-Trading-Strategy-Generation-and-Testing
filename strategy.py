#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: Uses weekly trend filter (price > EMA50 on weekly) with daily Camarilla R1/S1 breakouts
# and volume confirmation on 12h timeframe. Weekly trend ensures alignment with major market direction
# (works in bull/bear by following long-term trend), while Camarilla levels provide precise entry/exit
# zones. Volume confirmation filters false breakouts. Target: 15-25 trades/year to minimize fee drag.

timeframe = "12h"
name = "12h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day (using typical price)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # Where C = (H+L+C)/3 of previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price of previous day
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    # Range of previous day
    range_1d = high_1d - low_1d
    
    # Camarilla R1 and S1 levels
    camarilla_r1 = typical_price_1d + range_1d * 1.1 / 12
    camarilla_s1 = typical_price_1d - range_1d * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike detection: 1.5x average volume (2-period = 1 day on 12h chart)
    vol_ma = pd.Series(volume).rolling(window=2, min_periods=2).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 2)  # Ensure we have weekly EMA50 and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R1 with volume, and weekly trend is bullish (price > EMA50)
            if (high[i] > camarilla_r1_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i] and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1 with volume, and weekly trend is bearish (price < EMA50)
            elif (low[i] < camarilla_s1_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i] and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below Camarilla S1 (mean reversion to pivot)
            if low[i] < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above Camarilla R1 (mean reversion to pivot)
            if high[i] > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals