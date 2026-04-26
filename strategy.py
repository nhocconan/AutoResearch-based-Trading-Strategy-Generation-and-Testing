#!/usr/bin/env python3
"""
6h_Camarilla_R4S4_Breakout_12hTrend_VolumeFilter
Hypothesis: Camarilla pivot breakout at R4/S4 levels (strong continuation) with 12h EMA50 trend filter and volume confirmation.
Only go long when price breaks above R4 AND 12h EMA50 uptrend AND volume spike.
Only go short when price breaks below S4 AND 12h EMA50 downtrend AND volume spike.
Uses discrete position sizing (0.25) to minimize fee drag. Target: 12-37 trades/year per symbol.
Works in bull/bear via trend filter - only long in uptrend, short in downtrend.
Volume filter prevents low-conviction entries. R4/S4 breakouts indicate strong momentum continuation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter and Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels from previous 12h bar
    # Camarilla: R4 = C + ((H-L) * 1.1/2), S4 = C - ((H-L) * 1.1/2)
    # where C = (H+L+CLOSE)/3 of previous period
    typical_price_12h = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    camarilla_pivot = typical_price_12h.values
    camarilla_range = (high_12h - low_12h).values
    
    r4 = camarilla_pivot + (camarilla_range * 1.1 / 2)
    s4 = camarilla_pivot - (camarilla_range * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe (1-bar lag for completed 12h bar)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    
    # Volume spike detector (20-bar volume MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend filter from 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long: price breaks above R4, 12h uptrend, volume spike
            if close[i] > r4_aligned[i] and uptrend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4, 12h downtrend, volume spike
            elif close[i] < s4_aligned[i] and downtrend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below pivot OR trend changes
            if close[i] < camarilla_pivot[i] or not uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above pivot OR trend changes
            if close[i] > camarilla_pivot[i] or not downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R4S4_Breakout_12hTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0