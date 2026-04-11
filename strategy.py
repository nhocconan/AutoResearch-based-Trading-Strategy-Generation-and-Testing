#!/usr/bin/env python3
# 12h_1d_camarilla_breakout_volume_v1
# Strategy: 12h breakout at Camarilla levels calculated from 1d close, with volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Camarilla levels derived from daily price action provide strong support/resistance.
# Breakouts above resistance or below support with above-average volume capture momentum.
# Volume confirmation reduces false breakouts. Works in both bull and bear markets by
# following the direction of the breakout. Target: 12-37 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_volume_v1"
timeframe = "12h"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
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
    
    range_1d = high_1d - low_1d
    r4 = close_1d + range_1d * 1.1 / 2
    r3 = close_1d + range_1d * 1.1 / 4
    r2 = close_1d + range_1d * 1.1 / 6
    r1 = close_1d + range_1d * 1.1 / 12
    s1 = close_1d - range_1d * 1.1 / 12
    s2 = close_1d - range_1d * 1.1 / 6
    s3 = close_1d - range_1d * 1.1 / 4
    s4 = close_1d - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 12h (using previous 1d bar's levels)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: 1d volume > average
    vol_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_avg_20_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Current 1d volume (aligned)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, vol_1d)[i]
        vol_confirm = vol_1d_current > vol_avg_20_aligned[i]
        
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