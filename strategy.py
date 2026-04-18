#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_R1S1_Volume_Spike
12h strategy using daily Camarilla pivot levels R1/S1 with volume spike confirmation.
- Long: Close crosses above R1 + volume > 2.0x daily volume MA(30) + price above daily VWAP
- Short: Close crosses below S1 + volume > 2.0x daily volume MA(30) + price below daily VWAP
- Exit: Opposite cross of R1/S1 or VWAP cross in opposite direction
Designed for ~15-25 trades/year per symbol (60-100 total over 4 years)
Works in bull markets (breakout continuation) and bear markets (mean reversion at extremes)
"""

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
    
    # Get daily data for Camarilla pivots and VWAP
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels for each day
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R1 = Pivot + (Range * 1.1 / 12)
    # S1 = Pivot - (Range * 1.1 / 12)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1_1d = pivot_1d + (range_1d * 1.1 / 12.0)
    s1_1d = pivot_1d - (range_1d * 1.1 / 12.0)
    
    # Align Camarilla levels to 12h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Daily VWAP approximation using close (simplified)
    vwap_1d = (high_1d + low_1d + close_1d) / 3.0  # typical price as VWAP proxy
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Daily volume moving average (30-period)
    vol_ma_30 = pd.Series(volume_1d).rolling(window=30, min_periods=30).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # need enough for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vwap_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: volume spike > 2.0x average
        vol_spike = volume[i] > 2.0 * vol_ma_aligned[i]
        
        # Price relative to VWAP
        price_above_vwap = close[i] > vwap_aligned[i]
        price_below_vwap = close[i] < vwap_aligned[i]
        
        # Camarilla level crosses
        cross_above_r1 = close[i] > r1_aligned[i] and close[i-1] <= r1_aligned[i-1]
        cross_below_s1 = close[i] < s1_aligned[i] and close[i-1] >= s1_aligned[i-1]
        
        if position == 0:
            # Long: cross above R1 + volume spike + price above VWAP
            if cross_above_r1 and vol_spike and price_above_vwap:
                signals[i] = 0.25
                position = 1
            # Short: cross below S1 + volume spike + price below VWAP
            elif cross_below_s1 and vol_spike and price_below_vwap:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: cross below S1 or price below VWAP
            if cross_below_s1 or price_below_vwap:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: cross above R1 or price above VWAP
            if cross_above_r1 or price_above_vwap:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_Pivot_R1S1_Volume_Spike"
timeframe = "12h"
leverage = 1.0