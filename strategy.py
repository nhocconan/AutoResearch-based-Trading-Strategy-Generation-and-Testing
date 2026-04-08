#!/usr/bin/env python3
# 12h_1d_pivot_bounce_volume_v1
# Hypothesis: Trade bounces off 1d pivot points on 12h timeframe with volume confirmation.
# In both bull and bear markets, price often respects daily pivot levels (S1, S2, R1, R2).
# Long when price bounces off S1/S2 with volume surge and price above 1d VWAP (bullish bias).
# Short when price bounces off R1/R2 with volume surge and price below 1d VWAP (bearish bias).
# Uses 12h candles for entries to reduce trade frequency, targeting 12-37 trades/year.
# Pivot levels act as dynamic support/resistance that work in ranging and trending markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_pivot_bounce_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for pivot points and VWAP
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily pivot points: P = (H+L+C)/3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Support and resistance levels
    s1 = 2 * pp - high_1d
    s2 = pp - (high_1d - low_1d)
    r1 = 2 * pp - low_1d
    r2 = pp + (high_1d - low_1d)
    
    # Daily VWAP for bias
    vwap_num = (high_1d + low_1d + close_1d) * volume_1d
    vwap_den = volume_1d
    vwap = np.cumsum(vwap_num) / np.cumsum(vwap_den)
    
    # Align daily levels to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 20  # Ensure volume MA is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(vwap_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.8 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price breaks below S1 or reaches PP
            if close[i] < s1_aligned[i] or close[i] > pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above R1 or reaches PP
            if close[i] > r1_aligned[i] or close[i] < pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price bounces off S1/S2 with volume surge and price below VWAP (value area)
            near_support = (abs(close[i] - s1_aligned[i]) < 0.005 * close[i] or 
                           abs(close[i] - s2_aligned[i]) < 0.005 * close[i])
            if near_support and vol_surge and close[i] < vwap_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price bounces off R1/R2 with volume surge and price above VWAP
            elif (abs(close[i] - r1_aligned[i]) < 0.005 * close[i] or 
                  abs(close[i] - r2_aligned[i]) < 0.005 * close[i]):
                if vol_surge and close[i] > vwap_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals