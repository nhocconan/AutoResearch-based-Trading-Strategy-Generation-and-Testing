#!/usr/bin/env python3
"""
1d_1w_Pivot_R2S2_Breakout_Volume_ATRFilter
Hypothesis: Weekly Camarilla pivot levels R2/S2 act as strong support/resistance with breakout potential. Enter long when price breaks above R2 with volume confirmation, short when breaks below S2. Exit when price returns to weekly pivot P. Uses ATR for stoploss. Designed for low trade frequency (target: 10-30/year) to minimize fee drag in 1d timeframe. Works in both bull and bear markets by capturing momentum at key levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for Camarilla pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly Camarilla pivot levels
    # P = (H + L + C) / 3
    # Range = H - L
    # R2 = P + (Range * 1.1000/2) = P + (Range * 0.5500)
    # S2 = P - (Range * 1.1000/2) = P - (Range * 0.5500)
    # R4 = P + (Range * 1.5000)
    # S4 = P - (Range * 1.5000)
    P_weekly = (high_weekly + low_weekly + close_weekly) / 3.0
    range_weekly = high_weekly - low_weekly
    r2_weekly = P_weekly + (range_weekly * 0.5500)
    s2_weekly = P_weekly - (range_weekly * 0.5500)
    r4_weekly = P_weekly + (range_weekly * 1.5000)
    s4_weekly = P_weekly - (range_weekly * 1.5000)
    
    # Align weekly Camarilla levels to 1d timeframe
    r2_weekly_aligned = align_htf_to_ltf(prices, df_weekly, r2_weekly)
    s2_weekly_aligned = align_htf_to_ltf(prices, df_weekly, s2_weekly)
    r4_weekly_aligned = align_htf_to_ltf(prices, df_weekly, r4_weekly)
    s4_weekly_aligned = align_htf_to_ltf(prices, df_weekly, s4_weekly)
    P_weekly_aligned = align_htf_to_ltf(prices, df_weekly, P_weekly)
    
    # Main timeframe data (1d)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ATR for stoploss and volatility filter
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    high_close[0] = high[0] - close[0]  # first bar
    low_close[0] = low[0] - close[0]    # first bar
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = np.zeros_like(tr)
    for i in range(len(tr)):
        if i < 14:
            atr[i] = np.mean(tr[:i+1]) if i > 0 else tr[i]
        else:
            atr[i] = np.mean(tr[i-13:i+1])
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_avg = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            volume_avg[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
        else:
            volume_avg[i] = np.mean(volume[i-19:i+1])
    volume_filter = volume > (1.5 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if (np.isnan(r2_weekly_aligned[i]) or np.isnan(s2_weekly_aligned[i]) or 
            np.isnan(P_weekly_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r2 = r2_weekly_aligned[i]
        s2 = s2_weekly_aligned[i]
        p = P_weekly_aligned[i]
        r4 = r4_weekly_aligned[i]
        s4 = s4_weekly_aligned[i]
        vol_ok = volume_filter[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long breakout above R2 with volume
            if price > r2 and vol_ok:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short breakdown below S2 with volume
            elif price < s2 and vol_ok:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: return to P or stoploss
            if price < p or price < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: return to P or stoploss
            if price > p or price > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Pivot_R2S2_Breakout_Volume_ATRFilter"
timeframe = "1d"
leverage = 1.0