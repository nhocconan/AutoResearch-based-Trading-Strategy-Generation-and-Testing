#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R1S1_With_Volume_Spike_Trend_v1
Hypothesis: Buy when price breaks above Camarilla R1 with volume spike and above 1d EMA34 trend; sell when price breaks below S1 with volume spike and below 1d EMA34. Camarilla pivot levels (R1/S1) provide high-probability reversal/breakout levels in ranging markets. Volume spike confirms institutional participation, and 1d EMA34 ensures alignment with daily trend. Designed for low trade frequency (<30/year) to minimize fee drag while capturing trend continuations and reversals in both bull and bear markets.
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
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # 1d EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Camarilla pivot levels from previous day
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    # Range = high - low
    daily_range = high - low
    # Camarilla levels
    R1 = close + 1.1 * daily_range / 12
    S1 = close - 1.1 * daily_range / 12
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 40  # Need volume MA and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_aligned[i]) or 
            np.isnan(volume_spike[i]) or
            np.isnan(R1[i]) or
            np.isnan(S1[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_1d_val = ema_1d_aligned[i]
        vol_spike = volume_spike[i]
        r1 = R1[i]
        s1 = S1[i]
        
        if position == 0:
            # Long: price > R1 with volume spike and above 1d EMA34
            if price > r1 and vol_spike and price > ema_1d_val:
                signals[i] = 0.25
                position = 1
            # Short: price < S1 with volume spike and below 1d EMA34
            elif price < s1 and vol_spike and price < ema_1d_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price < S1 (reversion to mean) or below 1d EMA34
            if price < s1 or price < ema_1d_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price > R1 (reversion to mean) or above 1d EMA34
            if price > r1 or price > ema_1d_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_Pivot_R1S1_With_Volume_Spike_Trend_v1"
timeframe = "4h"
leverage = 1.0