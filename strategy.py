#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v4
Hypothesis: 4h Camarilla pivot (R1/S1) breakout with 1d EMA34 trend filter and volume spike.
Enters long when price breaks above R1, close above 1d EMA34, and volume spike.
Enters short when price breaks below S1, close below 1d EMA34, and volume spike.
Uses 4h primary timeframe targeting 20-50 trades/year (80-200 total over 4 years).
Camarilla levels provide precise intraday support/resistance, 1d EMA filters counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    pivot = (high + low + close) / 3.0
    range_val = high - low
    r1 = close + range_val * 1.1 / 12
    s1 = close - range_val * 1.1 / 12
    return r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 4h Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    r1_4h, s1_4h = calculate_camarilla(df_4h['high'], df_4h['low'], df_4h['close'])
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # Volume confirmation: volume > 2.0x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for 1d EMA, 20 for volume MA)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(r1_4h_aligned[i]) or 
            np.isnan(s1_4h_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above R1, close above 1d EMA34, volume spike
            if (close[i] > r1_4h_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, close below 1d EMA34, volume spike
            elif (close[i] < s1_4h_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below S1 OR close below 1d EMA34
            if (close[i] < s1_4h_aligned[i] or close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above R1 OR close above 1d EMA34
            if (close[i] > r1_4h_aligned[i] or close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v4"
timeframe = "4h"
leverage = 1.0