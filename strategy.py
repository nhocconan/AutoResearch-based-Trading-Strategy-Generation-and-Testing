#!/usr/bin/env python3
"""
6h_Camarilla_R4_S4_Breakout_1wTrend_Volume_v1
Hypothesis: Focus on breakouts at R4/S4 levels (stronger breakout signals) combined with weekly trend filter (1w EMA50) and volume confirmation (volume > 1.5x average). R4/S4 breakouts indicate stronger momentum and are less prone to false signals. Weekly trend filter ensures we only trade in the direction of the higher timeframe trend, reducing whipsaws in choppy markets. Volume confirmation adds conviction to breakouts. Designed for 10-25 trades/year per symbol with emphasis on quality.
"""
name = "6h_Camarilla_R4_S4_Breakout_1wTrend_Volume_v1"
timeframe = "6h"
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
    
    # Get 1d data for Camarilla pivot (R4/S4)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1-day OHLC for Camarilla pivot
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot levels calculation
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r4_1d = close_1d + (range_1d * 1.5)  # R4 = Close + Range * 1.5
    s4_1d = close_1d - (range_1d * 1.5)  # S4 = Close - Range * 1.5
    
    # Align Camarilla levels to 6h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 1-week EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter: current volume > 1.5 * 50-period average
    vol_avg = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup for weekly EMA50 and volume average
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_avg[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R4 + weekly uptrend + volume confirmation
            if (close[i] > r4_1d_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 + weekly downtrend + volume confirmation
            elif (close[i] < s4_1d_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to opposite Camarilla level (S4 for long, R4 for short)
            if position == 1:
                if close[i] <= s4_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] >= r4_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals