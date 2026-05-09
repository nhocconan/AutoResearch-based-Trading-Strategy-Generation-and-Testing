#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    """
    1h Camarilla pivot R1/S1 breakout with 4h trend filter and volume confirmation.
    - Long: Close breaks above R1 with volume > 1.5x average and price > 4h EMA(34)
    - Short: Close breaks below S1 with volume > 1.5x average and price < 4h EMA(34)
    - Exit: Opposite breakout or price crosses back through pivot point (PP)
    - Uses Camarilla levels from previous 1h session (derived from 4h data)
    - Session filter: 08-20 UTC
    - Target: 15-30 trades/year on 1h timeframe (60-120 total over 4 years)
    """
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla calculation and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 40:
        return np.zeros(n)
    
    # Calculate 4h EMA(34) for trend filter
    close_4h = pd.Series(df_4h['close'].values)
    ema34_4h = close_4h.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Calculate Camarilla levels from previous 4h session
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_vals = df_4h['close'].values
    
    # Calculate pivot point and Camarilla levels
    pp = (high_4h + low_4h + close_4h_vals) / 3
    range_4h = high_4h - low_4h
    r1 = pp + (range_4h * 1.1 / 12)
    s1 = pp - (range_4h * 1.1 / 12)
    
    # Align Camarilla levels to 1h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_4h, pp)
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (only trade during active hours)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema34_4h_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(vol_ma20[i]) or
            not (8 <= hours[i] <= 20)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20[i]
        
        if position == 0:
            # Long: Close breaks above R1 with volume confirmation and above 4h EMA trend
            if close[i] > r1_aligned[i] and vol_ok and close[i] > ema34_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: Close breaks below S1 with volume confirmation and below 4h EMA trend
            elif close[i] < s1_aligned[i] and vol_ok and close[i] < ema34_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: Close breaks below PP or opposite signal
            if close[i] < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Close breaks above PP or opposite signal
            if close[i] > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals