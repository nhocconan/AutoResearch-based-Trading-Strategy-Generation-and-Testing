#!/usr/bin/env python3
"""
4h_PivotPoint_R1S1_Breakout_Volume_ATRFilter
Hypothesis: Daily Pivot Points (R1/S1) act as key support/resistance levels. Breakouts above R1 with volume confirmation and ATR filter indicate strong momentum in trending markets, while failures to hold S1 indicate short opportunities. Designed for low trade frequency (target: 20-50/year) to minimize fee drag in 4h timeframe. Works in both bull and bear markets by using price action at key levels with volume confirmation and volatility filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for Pivot Points
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate daily Pivot Points
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    P = (high_daily + low_daily + close_daily) / 3.0
    r1_daily = 2 * P - low_daily
    s1_daily = 2 * P - high_daily
    
    # Align daily Pivot levels to 4h timeframe
    r1_daily_aligned = align_htf_to_ltf(prices, df_daily, r1_daily)
    s1_daily_aligned = align_htf_to_ltf(prices, df_daily, s1_daily)
    
    # Main timeframe data (4h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_avg = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 20:
            volume_avg[i] = np.mean(volume[i-20:i])
        else:
            volume_avg[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
    volume_filter = volume > (1.5 * volume_avg)
    
    # ATR(14) for volatility filter and stop reference
    tr1 = np.zeros_like(high)
    tr2 = np.zeros_like(high)
    tr3 = np.zeros_like(high)
    tr1[1:] = high[1:] - low[1:]
    tr2[1:] = np.abs(high[1:] - close[:-1])
    tr3[1:] = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.zeros_like(tr)
    for i in range(len(tr)):
        if i >= 14:
            atr[i] = np.mean(tr[i-14:i])
        else:
            atr[i] = np.mean(tr[:i+1]) if i > 0 else tr[i]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if (np.isnan(r1_daily_aligned[i]) or np.isnan(s1_daily_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r1 = r1_daily_aligned[i]
        s1 = s1_daily_aligned[i]
        vol_ok = volume_filter[i]
        atr_val = atr[i]
        
        if position == 0:
            # Breakout above R1 with volume confirmation
            if price > r1 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Breakdown below S1 with volume confirmation
            elif price < s1 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to S1 (failed breakout) or trailing stop
            if price < s1 or price < high[max(0, i-3):i+1].max() - 1.5 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to R1 (failed breakdown) or trailing stop
            if price > r1 or price > low[max(0, i-3):i+1].min() + 1.5 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_PivotPoint_R1S1_Breakout_Volume_ATRFilter"
timeframe = "4h"
leverage = 1.0