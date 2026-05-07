#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_12hEMA50_Trend_VolumeS
Hypothesis: Price breaking above Camarilla R1 or below S1 levels from the prior 12h period, combined with 12h EMA50 trend filter and volume confirmation, captures momentum moves. The 12h trend filter reduces false breakouts in choppy conditions, while the tighter R1/S1 levels increase trade frequency vs R3/S3 to reach optimal 20-50 trades/year on 4h.
"""
name = "4h_Camarilla_R1S1_Breakout_12hEMA50_Trend_VolumeS"
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
    
    # Get 12h data for Camarilla levels and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h Camarilla R1/S1 levels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    range_12h = high_12h - low_12h
    r1_12h = close_12h + 1.1666 * range_12h * 0.5 / 2  # R1 = C + 0.5*(H-L)*1.1666/2
    s1_12h = close_12h - 1.1666 * range_12h * 0.5 / 2  # S1 = C - 0.5*(H-L)*1.1666/2
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all to 4h timeframe
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + 12h uptrend + volume
            if close[i] > r1_12h_aligned[i] and close[i] > ema_50_12h_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + 12h downtrend + volume
            elif close[i] < s1_12h_aligned[i] and close[i] < ema_50_12h_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price crosses back through the opposite S1/R1 level
            if position == 1:
                if close[i] < s1_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > r1_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals