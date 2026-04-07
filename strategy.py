#!/usr/bin/env python3
"""
4h_camarilla_pivot_1d_trend_volume_v3
Hypothesis: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
combined with EMA trend filter and volume confirmation on 4h timeframe. 
In ranging markets, fade at R3/S3 with EMA filter; in trending markets, 
breakout at R4/S4. Volume confirmation reduces false signals. 
Works in both bull and bear markets by adapting to regime via EMA filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_1d_trend_volume_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily OHLC for pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    camarilla_r4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    camarilla_s4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    
    # Align daily levels to 4h timeframe
    r4_4h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_4h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_4h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    ema50_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 20-period volume average on 4h
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or 
            np.isnan(r4_4h[i]) or np.isnan(s4_4h[i]) or 
            np.isnan(ema50_4h[i]) or np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below S3 (mean reversion fail) OR 
            # price breaks above R4 and EMA turns down (breakout fail)
            if close[i] < s3_4h[i] or (close[i] > r4_4h[i] and close[i] < ema50_4h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above R3 (mean reversion fail) OR
            # price breaks below S4 and EMA turns up (breakout fail)
            if close[i] > r3_4h[i] or (close[i] < s4_4h[i] and close[i] > ema50_4h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Mean reversion longs at S3 in uptrend (price > EMA)
            if (close[i] <= s3_4h[i] and 
                vol_confirm and 
                close[i] > ema50_4h[i]):
                position = 1
                signals[i] = 0.25
            # Mean reversion shorts at R3 in downtrend (price < EMA)
            elif (close[i] >= r3_4h[i] and 
                  vol_confirm and 
                  close[i] < ema50_4h[i]):
                position = -1
                signals[i] = -0.25
            # Breakout longs at R4 in uptrend
            elif (close[i] >= r4_4h[i] and 
                  vol_confirm and 
                  close[i] > ema50_4h[i]):
                position = 1
                signals[i] = 0.25
            # Breakout shorts at S4 in downtrend
            elif (close[i] <= s4_4h[i] and 
                  vol_confirm and 
                  close[i] < ema50_4h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals