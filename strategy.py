#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_Bounce_v1
Hypothesis: Use daily Camarilla pivot levels with volume confirmation on 4h timeframe.
Long when price bounces off daily S3 with volume > 1.5x 20-period average,
short when price bounces off daily R3 with volume > 1.5x 20-period average.
Camarilla levels are widely watched by institutions; bounce strategy works in both
trending and ranging markets. Designed for low trade frequency (target: 75-200 total
over 4 years) to minimize fee drag. Works in bull via bounces off support,
in bear via bounces off resistance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Pivot_Bounce_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].iloc[-2] if len(df_1d) >= 2 else df_1d['high'].iloc[-1]
    prev_low = df_1d['low'].iloc[-2] if len(df_1d) >= 2 else df_1d['low'].iloc[-1]
    prev_close = df_1d['close'].iloc[-2] if len(df_1d) >= 2 else df_1d['close'].iloc[-1]
    
    # Calculate daily Camarilla pivot levels
    range_val = prev_high - prev_low
    if range_val <= 0:
        return np.zeros(n)
    
    # Camarilla levels
    camarilla_r3 = prev_close + range_val * 1.1 / 4  # R3 = Close + (Range * 1.1/4)
    camarilla_s3 = prev_close - range_val * 1.1 / 4  # S3 = Close - (Range * 1.1/4)
    camarilla_r4 = prev_close + range_val * 1.1 / 2  # R4 = Close + (Range * 1.1/2)
    camarilla_s4 = prev_close - range_val * 1.1 / 2  # S4 = Close - (Range * 1.1/2)
    
    # Align daily levels to 4h timeframe
    camarilla_r3_array = np.full(len(df_1d), camarilla_r3)
    camarilla_s3_array = np.full(len(df_1d), camarilla_s3)
    camarilla_r4_array = np.full(len(df_1d), camarilla_r4)
    camarilla_s4_array = np.full(len(df_1d), camarilla_s4)
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_array)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_array)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_array)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_array)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = volume_series / vol_ma
    vol_ratio = vol_ratio.fillna(1.0).values  # default to 1.0 if no MA
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Bounce conditions with volume filter
        long_bounce = (low[i] <= camarilla_s3_aligned[i] and 
                       close[i] > camarilla_s3_aligned[i] and
                       vol_ratio[i] > 1.5)
        short_bounce = (high[i] >= camarilla_r3_aligned[i] and 
                        close[i] < camarilla_r3_aligned[i] and
                        vol_ratio[i] > 1.5)
        
        # Exit conditions: opposite Camarilla level
        long_exit = close[i] >= camarilla_r3_aligned[i]
        short_exit = close[i] <= camarilla_s3_aligned[i]
        
        # Signal logic
        if long_bounce and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_bounce and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals