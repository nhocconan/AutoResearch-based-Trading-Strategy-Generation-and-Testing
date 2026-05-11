#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dTrend_VolumeS
Hypothesis: Uses Camarilla pivot levels from 1d timeframe for breakout signals,
filtered by 1d EMA trend and volume spikes. Camarilla R3/S3 levels represent
strong support/resistance where breakouts often lead to sustained moves.
Trades only in direction of 1d EMA34 trend to avoid counter-trend whipsaws.
Volume confirmation ensures breakouts have institutional interest.
Designed for low trade frequency (20-50/year) via strict entry conditions.
Works in both bull and bear markets by following higher-timeframe trend.
"""

name = "4h_Camarilla_R3S3_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    # Typical price
    typical = (high + low + close) / 3
    # Pivot point
    pivot = typical
    # Range
    range_val = high - low
    # Camarilla levels
    r4 = close + range_val * 1.500
    r3 = close + range_val * 1.250
    r2 = close + range_val * 1.166
    r1 = close + range_val * 1.083
    s1 = close - range_val * 1.083
    s2 = close - range_val * 1.166
    s3 = close - range_val * 1.250
    s4 = close - range_val * 1.500
    return r4, r3, r2, r1, pivot, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Daily Camarilla Pivots (R3/S3) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    r4_1d, r3_1d, r2_1d, r1_1d, pivot_1d, s1_1d, s2_1d, s3_1d, s4_1d = calculate_camarilla(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    
    # Align daily Camarilla to 4h timeframe
    r3_1d_4h = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_4h = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # --- Daily EMA34 for Trend Filter ---
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_4h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # --- Volume Spike Detection (20-period average on 4h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma, out=np.ones_like(volume), where=vol_ma!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_1d_4h[i]) or np.isnan(s3_1d_4h[i]) or 
            np.isnan(ema34_1d_4h[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: price breaks above R3 + uptrend + volume
            if (close[i] > r3_1d_4h[i] and 
                close[i] > ema34_1d_4h[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + downtrend + volume
            elif (close[i] < s3_1d_4h[i] and 
                  close[i] < ema34_1d_4h[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Camarilla level or trend reversal
            if position == 1:
                # Exit long: price breaks below S3 OR trend turns down
                if (close[i] < s3_1d_4h[i] or 
                    close[i] < ema34_1d_4h[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above R3 OR trend turns up
                if (close[i] > r3_1d_4h[i] or 
                    close[i] > ema34_1d_4h[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals