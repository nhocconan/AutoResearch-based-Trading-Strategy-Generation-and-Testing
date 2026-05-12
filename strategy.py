#!/usr/bin/env python3
name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h trend: EMA34
    df_4h = get_htf_data(prices, '4h')
    ema34_4h = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # 1d high/low for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R1, S1
    # R1 = close + (high - low) * 1.1 / 12
    # S1 = close - (high - low) * 1.1 / 12
    camarilla_range = (high_1d - low_1d) * 1.1 / 12
    r1 = close_1d + camarilla_range
    s1 = close_1d - camarilla_range
    
    # Align R1/S1 to 1h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume filter: 20-period EMA of volume
    volume_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # need enough data for EMA34 and volume EMA
    
    for i in range(start_idx, n):
        # Skip if 4h trend or volume data not ready
        if np.isnan(ema34_4h_aligned[i]) or np.isnan(volume_ema[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 + 4h uptrend + volume above average
            if (close[i] > r1_aligned[i] and 
                close[i] > ema34_4h_aligned[i] and 
                volume[i] > volume_ema[i]):
                signals[i] = 0.20
                position = 1
            # Short: break below S1 + 4h downtrend + volume above average
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema34_4h_aligned[i] and 
                  volume[i] > volume_ema[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long when price closes below S1 or 4h trend turns down
            if (close[i] < s1_aligned[i] or close[i] < ema34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short when price closes above R1 or 4h trend turns up
            if (close[i] > r1_aligned[i] or close[i] > ema34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals