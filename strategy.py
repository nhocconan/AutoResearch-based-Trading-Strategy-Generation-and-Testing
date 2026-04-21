#!/usr/bin/env python3
"""
4h_1d_Pivot_R2S2_Breakout_Volume_ATRFilter
Hypothesis: Daily pivot points R2/S2 are stronger support/resistance than R1/S1. Breakouts with volume confirmation and ATR stops capture significant moves. Designed for low trade frequency to minimize fee drift. Uses 4h primary timeframe with 1d pivot points.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data once for pivot points and ATR
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate daily True Range for ATR
    tr1 = np.abs(high_daily - low_daily)
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr1[0] = high_daily[0] - low_daily[0]
    tr2[0] = np.abs(high_daily[0] - close_daily[0])
    tr3[0] = np.abs(low_daily[0] - close_daily[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_daily = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily pivot points: P = (H+L+C)/3, R2 = P + (H-L), S2 = P - (H-L)
    pivot_daily = (high_daily + low_daily + close_daily) / 3.0
    r2_daily = pivot_daily + (high_daily - low_daily)
    s2_daily = pivot_daily - (high_daily - low_daily)
    
    # Align daily indicators to 4h timeframe
    atr_daily_aligned = align_htf_to_ltf(prices, df_daily, atr_daily)
    r2_daily_aligned = align_htf_to_ltf(prices, df_daily, r2_daily)
    s2_daily_aligned = align_htf_to_ltf(prices, df_daily, s2_daily)
    
    # Main timeframe data (4h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(atr_daily_aligned[i]) or np.isnan(r2_daily_aligned[i]) or np.isnan(s2_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        atr = atr_daily_aligned[i]
        r2 = r2_daily_aligned[i]
        s2 = s2_daily_aligned[i]
        vol_current = volume[i]
        
        # Volume filter: current volume > 2.0x 30-period average
        vol_ma = np.mean(volume[max(0, i-30):i]) if i >= 30 else volume[i]
        vol_ok = vol_current > 2.0 * vol_ma
        
        if position == 0:
            # Long breakout: price breaks above R2 with volume confirmation
            if price > r2 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S2 with volume confirmation
            elif price < s2 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S2 (failed breakout) or ATR-based stop
            if price < s2 or (i > 0 and close[i-1] > s2 and price < close[i-1] - 1.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R2 (failed breakdown) or ATR-based stop
            if price > r2 or (i > 0 and close[i-1] < r2 and price > close[i-1] + 1.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_Pivot_R2S2_Breakout_Volume_ATRFilter"
timeframe = "4h"
leverage = 1.0