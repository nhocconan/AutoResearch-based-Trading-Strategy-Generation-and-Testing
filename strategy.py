#!/usr/bin/env python3
# 4h_12h_camarilla_breakout_volume_v1
# Strategy: 4h breakout at Camarilla levels calculated from 12h close, with volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla levels derived from 12h price action provide strong support/resistance.
# Breakouts above resistance or below support with above-average volume capture momentum.
# Volume confirmation reduces false breakouts. Works in both bull and bear markets by
# following the direction of the breakout. Target: 20-40 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_breakout_volume_v1"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from 12h OHLC
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = Close + Range * 1.1/2
    # R3 = Close + Range * 1.1/4
    # R2 = Close + Range * 1.1/6
    # R1 = Close + Range * 1.1/12
    # S1 = Close - Range * 1.1/12
    # S2 = Close - Range * 1.1/6
    # S3 = Close - Range * 1.1/4
    # S4 = Close - Range * 1.1/2
    # where Range = High - Low
    
    range_12h = high_12h - low_12h
    r4 = close_12h + range_12h * 1.1 / 2
    r3 = close_12h + range_12h * 1.1 / 4
    r2 = close_12h + range_12h * 1.1 / 6
    r1 = close_12h + range_12h * 1.1 / 12
    s1 = close_12h - range_12h * 1.1 / 12
    s2 = close_12h - range_12h * 1.1 / 6
    s3 = close_12h - range_12h * 1.1 / 4
    s4 = close_12h - range_12h * 1.1 / 2
    
    # Align Camarilla levels to 4h (using previous 12h bar's levels)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    r2_aligned = align_htf_to_ltf(prices, df_12h, r2)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    s2_aligned = align_htf_to_ltf(prices, df_12h, s2)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    
    # Volume confirmation: 12h volume > average
    vol_12h = df_12h['volume'].values
    vol_avg_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_avg_20_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Current 12h volume (aligned)
        vol_12h_current = align_htf_to_ltf(prices, df_12h, vol_12h)[i]
        vol_confirm = vol_12h_current > vol_avg_20_aligned[i]
        
        # Breakout conditions using Camarilla levels
        breakout_above_r1 = close[i] > r1_aligned[i-1]  # break above R1
        breakout_below_s1 = close[i] < s1_aligned[i-1]  # break below S1
        
        long_signal = breakout_above_r1 and vol_confirm
        short_signal = breakout_below_s1 and vol_confirm
        
        # Exit conditions: opposite breakout or volume failure
        long_exit = close[i] < s1_aligned[i-1] or not vol_confirm
        short_exit = close[i] > r1_aligned[i-1] or not vol_confirm
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals