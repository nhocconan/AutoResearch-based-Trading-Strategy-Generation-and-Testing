#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    """
    12h Camarilla R1/S1 breakout with 1w trend filter and volume confirmation.
    - Uses 1w EMA50 for trend direction
    - R1/S1 breakouts for momentum entries
    - Volume spike filter to avoid false breakouts
    - Target: 12-37 trades/year on 12h timeframe
    """
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla R1 and S1 from previous 1d period's OHLC
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    prev_range = prev_high_1d - prev_low_1d
    pivot = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    r1 = pivot + 1.1 * prev_range * 1.05  # R1 = pivot + 1.1 * range * 1.05
    s1 = pivot - 1.1 * prev_range * 1.05  # S1 = pivot - 1.1 * range * 1.05
    
    # Align Camarilla levels to 12h
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1w EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume spike detection (10-period for 12h)
    vol_avg = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or np.isnan(ema50_12h[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.8 x 10-period average
        vol_spike = volume[i] > vol_avg[i] * 1.8
        
        if position == 0:
            # Long: Break above Camarilla R1 with uptrend on 1w, volume spike
            if (close[i] > r1_12h[i] and close[i] > ema50_12h[i] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S1 with downtrend on 1w, volume spike
            elif (close[i] < s1_12h[i] and close[i] < ema50_12h[i] and vol_spike):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below EMA50
            if close[i] < ema50_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above EMA50
            if close[i] > ema50_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals