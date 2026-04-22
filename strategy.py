#!/usr/bin/env python3

"""
Hypothesis: 4-hour Camarilla pivot level touch with 1-day EMA34 trend filter and volume confirmation.
Trade long when price touches Camarilla S1/S2 in uptrend (price > EMA34) with volume spike.
Trade short when price touches Camarilla R1/R2 in downtrend (price < EMA34) with volume spike.
Uses Camarilla levels from daily timeframe for institutional support/resistance, EMA34 for trend,
and volume spikes to confirm institutional participation. Designed for low trade frequency (20-40 trades/year)
by requiring confluence of trend, level touch, and volume. Works in both bull and bear markets by
following the trend direction from EMA34, which adapts to market conditions.
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
    
    # Load 1d data for Camarilla and EMA34 - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d for trend
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla levels from previous day
    high_prev = np.roll(df_1d['high'].values, 1)
    low_prev = np.roll(df_1d['low'].values, 1)
    close_prev = np.roll(df_1d['close'].values, 1)
    high_prev[0] = high_prev[1] if len(high_prev) > 1 else high_prev[0]
    low_prev[0] = low_prev[1] if len(low_prev) > 1 else low_prev[0]
    close_prev[0] = close_prev[1] if len(close_prev) > 1 else close_prev[0]
    
    # Camarilla levels: Close + (High-Low) * multiplier
    rang = high_prev - low_prev
    r1 = close_prev + rang * 1.1 / 12
    r2 = close_prev + rang * 1.1 / 6
    s1 = close_prev - rang * 1.1 / 12
    s2 = close_prev - rang * 1.1 / 6
    
    # Align to 4h
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: price touches S1 or S2 in uptrend (price > EMA34) with volume spike
            if (close[i] <= s1_aligned[i] * 1.002 or close[i] <= s2_aligned[i] * 1.002) and \
               close[i] > ema34_1d_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price touches R1 or R2 in downtrend (price < EMA34) with volume spike
            elif (close[i] >= r1_aligned[i] * 0.998 or close[i] >= r2_aligned[i] * 0.998) and \
                 close[i] < ema34_1d_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price reaches opposite Camarilla level or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price reaches R1 or trend turns down
                if close[i] >= r1_aligned[i] * 0.998 or close[i] < ema34_1d_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reaches S1 or trend turns up
                if close[i] <= s1_aligned[i] * 1.002 or close[i] > ema34_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_Pivot_Touch_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0