#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_S1_S3_Breakout_1D_Trend_Force_v2
Hypothesis: Use daily Camarilla pivot levels S1 and S3 as breakout levels for 4h timeframe. Go long when price breaks above S1 with volume confirmation and 1d EMA34 uptrend, short when price breaks below S3 with volume confirmation and 1d EMA34 downtrend. Camarilla levels provide high-probability reversal/breakout points. Works in both bull (catching breakouts) and bear (catching reversals) markets. Designed for 4h timeframe with moderate trade frequency (target: 20-50/year).
"""

name = "4h_Camarilla_Pivot_S1_S3_Breakout_1D_Trend_Force_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pivot = (high_1d + low_1d + close_1d) / 3
    # Camarilla levels
    s1 = close_1d - (high_1d - low_1d) * 1.06 / 12
    s2 = close_1d - (high_1d - low_1d) * 1.06 / 6
    s3 = close_1d - (high_1d - low_1d) * 1.06 / 4
    r1 = close_1d + (high_1d - low_1d) * 1.06 / 12
    r2 = close_1d + (high_1d - low_1d) * 1.06 / 6
    r3 = close_1d + (high_1d - low_1d) * 1.06 / 4
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all daily data to 4h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume average (20-period) for volume spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup for EMA and volume MA
        # Skip if any required data is NaN
        if (np.isnan(s1_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.3x 20-period average
        vol_spike = volume[i] > 1.3 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above S1 with volume spike and 1d EMA34 uptrend
            if close[i] > s1_aligned[i] and vol_spike and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with volume spike and 1d EMA34 downtrend
            elif close[i] < s3_aligned[i] and vol_spike and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 or 1d EMA34 turns down
            if close[i] < s3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above S1 or 1d EMA34 turns up
            if close[i] > s1_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals