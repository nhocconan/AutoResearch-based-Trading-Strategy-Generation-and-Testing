#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA50 trend filter and volume spike (>2.0x 20-period MA) on 12h timeframe.
Designed for low trade frequency (~12-37/year) to minimize fee drag while capturing strong trending moves aligned with daily trend.
Uses 1d HTF for trend and pivot levels to ensure proper multi-timeframe alignment without look-ahead.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA trend and Camarilla)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d OHLC for Camarilla pivot calculation (based on previous 1d bar) ===
    df_1d_open = df_1d['open'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    range_1d = df_1d_high - df_1d_low
    r1_1d = df_1d_close + 0.275 * range_1d
    s1_1d = df_1d_close - 0.275 * range_1d
    
    # Align 1d Camarilla levels to 12h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 1d EMA50 for trend filter ===
    ema_50_1d = pd.Series(df_1d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 12h Volume spike filter (2.0x 20-period MA) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) 
            or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume_now = volume[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        ema_50 = ema_50_1d_aligned[i]
        vol_avg = vol_ma[i]
        
        # Volume spike: current volume > 2.0x average
        volume_spike = volume_now > 2.0 * vol_avg
        
        if position == 0:
            # Enter only with volume spike and trend alignment
            long_condition = (price > r1) and (price > ema_50) and volume_spike
            short_condition = (price < s1) and (price < ema_50) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit on trend reversal or volume spike in opposite direction
            if price < ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit on trend reversal or volume spike in opposite direction
            if price > ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0