#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    """
    4h Camarilla R1/S1 breakout with 12h trend filter and volume confirmation.
    - Uses 12h EMA50 for trend direction
    - R1/S1 breakouts for momentum entries
    - Volume spike filter to avoid false breakouts
    - Target: 20-50 trades/year on 4h timeframe
    """
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivots and trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla R1 and S1 from previous period's OHLC
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    prev_close_12h = np.roll(close_12h, 1)
    prev_high_12h[0] = np.nan
    prev_low_12h[0] = np.nan
    prev_close_12h[0] = np.nan
    
    prev_range = prev_high_12h - prev_low_12h
    pivot = (prev_high_12h + prev_low_12h + prev_close_12h) / 3
    r1 = pivot + 1.1 * prev_range * 1.05  # R1 = pivot + 1.1 * range * 1.05
    s1 = pivot - 1.1 * prev_range * 1.05  # S1 = pivot - 1.1 * range * 1.05
    
    # Align Camarilla levels to 4h
    r1_4h = align_htf_to_ltf(prices, df_12h, r1)
    s1_4h = align_htf_to_ltf(prices, df_12h, s1)
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume spike detection (10-period for 4h)
    vol_avg = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or np.isnan(ema50_4h[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.8 x 10-period average
        vol_spike = volume[i] > vol_avg[i] * 1.8
        
        if position == 0:
            # Long: Break above Camarilla R1 with uptrend on 12h, volume spike
            if (close[i] > r1_4h[i] and close[i] > ema50_4h[i] and vol_spike):
                signals[i] = 0.30
                position = 1
            # Short: Break below Camarilla S1 with downtrend on 12h, volume spike
            elif (close[i] < s1_4h[i] and close[i] < ema50_4h[i] and vol_spike):
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below EMA50
            if close[i] < ema50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit short: Price rises back above EMA50
            if close[i] > ema50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals