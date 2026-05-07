#!/usr/bin/env python3
"""
4H_Camarilla_R1S1_Breakout_1dEMA34_Trend_Filter_v1
Hypothesis: Long when price breaks above 4h R1 level with 1d EMA34 trend up and volume confirmation; 
short when price breaks below 4h S1 level with 1d EMA34 trend down and volume confirmation.
Uses 4h Camarilla levels for precise entry, 1d EMA for trend filter, and volume spike for confirmation.
Designed to work in both bull and bear markets by following higher timeframe trend.
"""
name = "4H_Camarilla_R1S1_Breakout_1dEMA34_Trend_Filter_v1"
timeframe = "4h"
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
    
    # Get 4h data for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (R1, S1)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    pivot_4h = (high_4h + low_4h + close_4h) / 3
    range_4h = high_4h - low_4h
    r1_4h = pivot_4h + (range_4h * 1.1 / 12)
    s1_4h = pivot_4h - (range_4h * 1.1 / 12)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # Get 1d data for EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume filter: current volume > 1.5 * 20-period average volume
    vol_series = pd.Series(volume)
    vol_avg = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1, 1d EMA34 up, volume confirmation
            if (close[i] > r1_4h_aligned[i] and close[i-1] <= r1_4h_aligned[i-1] and
                ema_34_aligned[i] > ema_34_aligned[i-1] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, 1d EMA34 down, volume confirmation
            elif (close[i] < s1_4h_aligned[i] and close[i-1] >= s1_4h_aligned[i-1] and
                  ema_34_aligned[i] < ema_34_aligned[i-1] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to opposite S1/R1 level or trend changes
            if position == 1:
                if (close[i] < s1_4h_aligned[i] or ema_34_aligned[i] < ema_34_aligned[i-1]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if (close[i] > r1_4h_aligned[i] or ema_34_aligned[i] > ema_34_aligned[i-1]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals