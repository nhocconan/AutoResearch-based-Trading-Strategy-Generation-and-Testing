#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_Breakout_12hTrend_Volume
Hypothesis: Use 12h Camarilla pivot levels (R1/S1) with breakout logic, confirmed by 12h EMA50 trend and volume spikes. 
Camarilla levels provide precise support/resistance in ranging markets, while EMA50 filters direction.
Volume spike confirms breakout strength. Designed for 4h timeframe with 12h HTF to limit trades (target 20-50/year)
and work in both bull (buy R1 breakouts in uptrend) and bear (sell S1 breakdowns in downtrend) markets.
"""

name = "4h_Camarilla_Pivot_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Pivot point = (H + L + C) / 3
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    # Range = H - L
    range_12h = high_12h - low_12h
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    r1_12h = close_12h + (range_12h * 1.1 / 12)
    s1_12h = close_12h - (range_12h * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    # Get 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate volume average (20-period) for volume spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5x 20-period average
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above R1 with volume spike and above EMA50 (uptrend)
            if close[i] > r1_12h_aligned[i] and vol_spike and close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume spike and below EMA50 (downtrend)
            elif close[i] < s1_12h_aligned[i] and vol_spike and close[i] < ema_50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 or volume dries up
            if close[i] < s1_12h_aligned[i] or volume[i] < vol_ma_20[i] * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 or volume dries up
            if close[i] > r1_12h_aligned[i] or volume[i] < vol_ma_20[i] * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals