#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout + volume confirmation on 1h timeframe.
Long when price breaks above R1 AND volume > 1.3x 20-period average.
Short when price breaks below S1 AND volume > 1.3x 20-period average.
Exit when price reverts to pivot point (PP) or opposite breakout occurs.
Uses 4h for Camarilla calculation (structure) and 1h only for entry timing.
Target: 60-120 total trades over 4 years (15-30/year). Camarilla pivots work well in ranging/volatile markets.
Volume confirmation reduces false breakouts. Works in both bull (captures rallies) and bear (captures drops).
"""

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
    
    # Get 4h data for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels on 4h timeframe (previous day's OHLC)
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12, PP = (H+L+C)/3
    # Using rolling window of 1 (previous completed 4h bar) - but we need to shift by 1 to avoid look-ahead
    high_4h_series = pd.Series(high_4h)
    low_4h_series = pd.Series(low_4h)
    close_4h_series = pd.Series(close_4h)
    
    # Previous completed bar's OHLC (shift by 1 to avoid look-ahead)
    prev_high = high_4h_series.shift(1)
    prev_low = low_4h_series.shift(1)
    prev_close = close_4h_series.shift(1)
    
    # Camarilla calculations
    rang = prev_high - prev_low
    R1 = prev_close + rang * 1.1 / 12
    S1 = prev_close - rang * 1.1 / 12
    PP = (prev_high + prev_low + prev_close) / 3
    
    # Convert to arrays and handle NaN from shift
    R1 = R1.values
    S1 = S1.values
    PP = PP.values
    
    # Volume average (20-period) on 4h
    volume_4h = df_4h['volume'].values
    volume_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align 4h indicators to 1h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_4h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_4h, S1)
    PP_aligned = align_htf_to_ltf(prices, df_4h, PP)
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(PP_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        r1 = R1_aligned[i]
        s1 = S1_aligned[i]
        pp = PP_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > R1 AND volume > 1.3x avg
            if price > r1 and vol > 1.3 * vol_ma:
                signals[i] = 0.20
                position = 1
            # Short: price < S1 AND volume > 1.3x avg
            elif price < s1 and vol > 1.3 * vol_ma:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price < PP OR short signal triggers
            if price < pp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price > PP OR long signal triggers
            if price > pp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "4h_Camarilla_R1S1_Volume_Breakout"
timeframe = "1h"
leverage = 1.0