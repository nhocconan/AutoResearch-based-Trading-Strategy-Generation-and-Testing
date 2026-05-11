#!/usr/bin/env python3
name = "1d_Weekly_Camarilla_R4_S4_Breakout_1wTrend_Volume"
timeframe = "1d"
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
    
    # 1w data for Camarilla pivot and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly Camarilla pivot levels (R4, S4)
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    r4_1w = close_1w + range_1w * 1.5
    s4_1w = close_1w - range_1w * 1.5
    
    # Align weekly R4 and S4 to daily
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Weekly EMA8 for trend filter
    ema8_1w = pd.Series(close_1w).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema8_1w_aligned = align_htf_to_ltf(prices, df_1w, ema8_1w)
    
    # Volume spike detection: current volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema8_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R4 with volume spike and uptrend (EMA8 rising)
            if (close[i] > r4_aligned[i] and 
                volume_spike[i] and
                ema8_1w_aligned[i] > ema8_1w_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 with volume spike and downtrend (EMA8 falling)
            elif (close[i] < s4_aligned[i] and 
                  volume_spike[i] and
                  ema8_1w_aligned[i] < ema8_1w_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below R4 or trend reverses
            if (close[i] < r4_aligned[i] or 
                ema8_1w_aligned[i] < ema8_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above S4 or trend reverses
            if (close[i] > s4_aligned[i] or 
                ema8_1w_aligned[i] > ema8_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals