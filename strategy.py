#!/usr/bin/env python3
"""
4h_1d_Camarilla_R1S1_Breakout_Volume_ATRFilter
Hypothesis: Daily Camarilla pivot levels R1/S1 act as reversal zones with volume confirmation, while R4/S4 indicate breakout strength. Uses 4h timeframe for optimal trade frequency (target: 20-50 trades/year) and includes ATR-based stoploss to manage risk. Designed to work in both bull and bear markets by fading extremes and catching breakouts.
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
    
    # ATR for volatility filter and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = np.zeros_like(close)
    for i in range(len(close)):
        if i < 14:
            atr[i] = np.mean(tr[:i+1]) if i > 0 else tr[i]
        else:
            atr[i] = np.mean(tr[i-13:i+1])
    
    # Volume filter: current volume > 1.3x 20-period average
    volume_avg = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            volume_avg[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
        else:
            volume_avg[i] = np.mean(volume[i-20:i+1])
    volume_filter = volume > (1.3 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if (np.isnan(r1_daily_aligned[i]) or np.isnan(s1_daily_aligned[i]) or 
            np.isnan(r4_daily_aligned[i]) or np.isnan(s4_daily_aligned[i]) or
            np.isnan(atr[i])):
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
        
        if position == 0:
            # Fade at R1/S1: mean reversion from extreme levels
            # Long: price rejects S1 with volume confirmation
            if price > s1 and price < (s1 + (r1 - s1) * 0.4) and vol_ok:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price rejects R1 with volume confirmation
            elif price < r1 and price > (r1 - (r1 - s1) * 0.4) and vol_ok:
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
            # Long exit conditions
            exit_signal = False
            # Mean reversion: return to S1
            if price < s1:
                exit_signal = True
            # Failed breakout: return below R4
            elif price < r4:
                exit_signal = True
            # Stoploss: 2.5 * ATR below entry
            elif price < entry_price - 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit conditions
            exit_signal = False
            # Mean reversion: return to R1
            if price > r1:
                exit_signal = True
            # Failed breakdown: return above S4
            elif price > s4:
                exit_signal = True
            # Stoploss: 2.5 * ATR above entry
            elif price > entry_price + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_Camarilla_R1S1_Breakout_Volume_ATRFilter"
timeframe = "4h"
leverage = 1.0