#!/usr/bin/env python3
"""
12h_1d_Camarilla_R1S1_Breakout_Volume_ATRFilter
Hypothesis: Camarilla pivot levels (R1/S1) from daily timeframe combined with volume confirmation and ATR-based trailing stop on 12h timeframe will capture strong intraday moves in both bull and bear markets. The daily pivot provides institutional reference levels, volume confirms institutional participation, and ATR stop manages risk. Designed for low trade frequency (target: 12-37/year) to minimize fee drag in 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for Camarilla pivot calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (R1, S1) from daily OHLC
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Typical price for pivot calculation
    typical_price = (high_daily + low_daily + close_daily) / 3.0
    # Camarilla pivot formula
    pivot = (high_daily + low_daily + close_daily) / 3.0
    range_daily = high_daily - low_daily
    # R1 = Close + (Range * 1.1/12)
    r1 = close_daily + (range_daily * 1.1 / 12)
    # S1 = Close - (Range * 1.1/12)
    s1 = close_daily - (range_daily * 1.1 / 12)
    
    # Align daily R1/S1 to 12h timeframe (wait for daily close)
    r1_aligned = align_htf_to_ltf(prices, df_daily, r1)
    s1_aligned = align_htf_to_ltf(prices, df_daily, s1)
    
    # Main timeframe data (12h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ATR for stoploss and filtering (20-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = np.zeros_like(close)
    for i in range(len(tr)):
        if i < 20:
            atr[i] = np.mean(tr[:i+1])
        else:
            atr[i] = np.mean(tr[i-20:i])
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_avg = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 20:
            volume_avg[i] = np.mean(volume[i-20:i])
        else:
            volume_avg[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
    volume_filter = volume > (1.5 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):  # Start after ATR/volume warmup
        # Skip if NaN in critical values
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        vol_ok = volume_filter[i]
        atr_val = atr[i]
        
        # Stoploss: 2.0 * ATR from entry
        if position == 1 and price < entry_price - 2.0 * atr_val:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and price > entry_price + 2.0 * atr_val:
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume
            if price > r1_level and vol_ok:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S1 with volume
            elif price < s1_level and vol_ok:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price returns to S1 (mean reversion) or breaks below S1 (failed breakout)
            if price < s1_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to R1 (mean reversion) or breaks above R1 (failed breakdown)
            if price > r1_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Camarilla_R1S1_Breakout_Volume_ATRFilter"
timeframe = "12h"
leverage = 1.0