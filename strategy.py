#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Volume_EMA34
4h strategy using daily Camarilla pivot levels (R1/S1) with volume confirmation and EMA34 trend filter.
- Long: Close crosses above R1 + volume > 1.5x daily avg + EMA34 rising
- Short: Close crosses below S1 + volume > 1.5x daily avg + EMA34 falling
- Exit: Opposite cross or EMA34 direction change
Designed for ~25-40 trades/year per symbol (100-160 total over 4 years)
Works in bull markets (breakout continuation) and bear markets (breakdown continuation)
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
    
    # Get daily data for Camarilla pivot levels and EMA34
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels (based on previous day)
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align Camarilla levels to 4h (use previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Daily volume average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # need enough for EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions (EMA34 slope)
        if i >= 1:
            ema_rising = ema_34_aligned[i] > ema_34_aligned[i-1]
            ema_falling = ema_34_aligned[i] < ema_34_aligned[i-1]
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # Cross conditions
        cross_above_r1 = close[i] > r1_aligned[i] and close[i-1] <= r1_aligned[i-1]
        cross_below_s1 = close[i] < s1_aligned[i] and close[i-1] >= s1_aligned[i-1]
        
        if position == 0:
            # Long: EMA34 rising + volume + cross above R1
            if ema_rising and vol_confirm and cross_above_r1:
                signals[i] = 0.25
                position = 1
            # Short: EMA34 falling + volume + cross below S1
            elif ema_falling and vol_confirm and cross_below_s1:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: EMA34 falling or cross below S1
            if not ema_rising or cross_below_s1:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: EMA34 rising or cross above R1
            if not ema_falling or cross_above_r1:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Volume_EMA34"
timeframe = "4h"
leverage = 1.0