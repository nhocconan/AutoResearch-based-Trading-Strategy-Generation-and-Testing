#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_Volume
Hypothesis: On 1h timeframe, price breaking above Camarilla R1 or below S1 levels from the prior 4h period, combined with 4h EMA20 trend filter and volume confirmation, captures momentum moves. The 4h trend filter ensures alignment with higher timeframe momentum, reducing false breakouts. Volume confirmation filters low-conviction moves. Designed for 15-37 trades/year on 1h timeframe with strict entry conditions to avoid fee drag.
"""
name = "1h_Camarilla_R1S1_Breakout_4hTrend_Volume"
timeframe = "1h"
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
    
    # Get 4h data for Camarilla levels and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h Camarilla R1/S1 levels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    range_4h = high_4h - low_4h
    r1_4h = close_4h + 1.1666 * range_4h * 0.5 / 2
    s1_4h = close_4h - 1.1666 * range_4h * 0.5 / 2
    
    # 4h EMA20 for trend filter
    ema_20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all to 1h timeframe
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) or 
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + 4h uptrend + volume
            if close[i] > r1_4h_aligned[i] and close[i] > ema_20_4h_aligned[i] and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 + 4h downtrend + volume
            elif close[i] < s1_4h_aligned[i] and close[i] < ema_20_4h_aligned[i] and volume_filter[i]:
                signals[i] = -0.20
                position = -1
        elif position != 0:
            # Exit: price crosses back through the opposite S1/R1 level
            if position == 1:
                if close[i] < s1_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if close[i] > r1_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals