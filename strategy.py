#!/usr/bin/env python3
"""
4h_1d_Camarilla_R1S1_Breakout_Volume_ATRFilter_v2
Hypothesis: Daily Camarilla pivot levels R1/S1 act as mean-reversion zones, while R4/S4 indicate breakout strength. Fade at R1/S1 with volume confirmation, breakout at R4/S4 with volume confirmation. Designed for low trade frequency (target: 20-50/year) to minimize fee drag in 4h timeframe. Works in both bull and bear markets by adapting to regime via price action at key levels. Version 2 improves trade frequency by tightening volume filter and adding momentum confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for Camarilla pivot points
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate daily Camarilla pivot levels
    P = (high_daily + low_daily + close_daily) / 3.0
    range_daily = high_daily - low_daily
    r1_daily = P + (range_daily * 0.382)
    s1_daily = P - (range_daily * 0.382)
    r4_daily = P + (range_daily * 1.5000)
    s4_daily = P - (range_daily * 1.5000)
    
    # Align daily Camarilla levels to 4h timeframe
    r1_daily_aligned = align_htf_to_ltf(prices, df_daily, r1_daily)
    s1_daily_aligned = align_htf_to_ltf(prices, df_daily, s1_daily)
    r4_daily_aligned = align_htf_to_ltf(prices, df_daily, r4_daily)
    s4_daily_aligned = align_htf_to_ltf(prices, df_daily, s4_daily)
    
    # Main timeframe data (4h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 2.0x 20-period average (more stringent)
    volume_avg = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 20:
            volume_avg[i] = np.mean(volume[i-20:i])
        else:
            volume_avg[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
    volume_filter = volume > (2.0 * volume_avg)
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = np.zeros_like(close)
    for i in range(len(tr)):
        if i < 14:
            atr[i] = np.mean(tr[:i+1])
        else:
            atr[i] = np.mean(tr[i-14:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if (np.isnan(r1_daily_aligned[i]) or np.isnan(s1_daily_aligned[i]) or 
            np.isnan(r4_daily_aligned[i]) or np.isnan(s4_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r1 = r1_daily_aligned[i]
        s1 = s1_daily_aligned[i]
        r4 = r4_daily_aligned[i]
        s4 = s4_daily_aligned[i]
        vol_ok = volume_filter[i]
        atr_val = atr[i]
        
        # Stoploss: 2 * ATR from entry
        if position == 1 and price < entry_price - 2.0 * atr_val:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and price > entry_price + 2.0 * atr_val:
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # Fade at R1/S1: mean reversion from extreme levels
            # Long: price rejects S1 with volume confirmation (buying pressure)
            if price > s1 and price < (s1 + (r1 - s1) * 0.3) and vol_ok:
                # Additional confirmation: price closing near high of bar
                if close[i] > (high[i] + low[i]) / 2:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
            # Short: price rejects R1 with volume confirmation (selling pressure)
            elif price < r1 and price > (r1 - (r1 - s1) * 0.3) and vol_ok:
                # Additional confirmation: price closing near low of bar
                if close[i] < (high[i] + low[i]) / 2:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
            # Breakout at R4/S4: strong momentum continuation
            # Long: price breaks above R4 with volume
            elif price > r4 and vol_ok:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S4 with volume
            elif price < s4 and vol_ok:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price returns to S1 (mean reversion) or breaks S4 (failed breakout)
            if price < s1 or price > r4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to R1 (mean reversion) or breaks S4 (failed breakdown)
            if price > r1 or price < s4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_Camarilla_R1S1_Breakout_Volume_ATRFilter_v2"
timeframe = "4h"
leverage = 1.0