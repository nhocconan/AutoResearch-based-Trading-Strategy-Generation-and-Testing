#!/usr/bin/env python3
"""
12h_Pivot_R1S1_Breakout_Volume_ADX_Filter
Hypothesis: 12h Camarilla R1/S1 breakout with volume confirmation and ADX trend filter
- Camarilla levels from prior 1d provide statistically significant support/resistance
- ADX > 25 filters for trending markets to avoid false breakouts in chop
- Volume confirmation ensures institutional participation
- Designed for 12h timeframe targeting 50-150 total trades over 4 years (12-37/year)
- Works in bull/bear via ADX trend filter (avoids range-bound false signals)
"""

name = "12h_Pivot_R1S1_Breakout_Volume_ADX_Filter"
timeframe = "12h"
leverage = 1.0

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
    
    # Wilder smoothing helper
    def WilderSmooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        alpha = 1.0 / period
        result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = result[i-1] + alpha * (data[i] - result[i-1])
            else:
                result[i] = np.nan
        return result
    
    # ADX(14) for trend strength filter - calculated on 12h data
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # True Range
    tr1 = df_12h['high'] - df_12h['low']
    tr2 = np.abs(df_12h['high'] - np.roll(df_12h['close'], 1))
    tr3 = np.abs(df_12h['low'] - np.roll(df_12h['close'], 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr.iloc[0] = tr1.iloc[0]
    
    # Directional Movement
    dm_plus = np.where((df_12h['high'] - np.roll(df_12h['high'], 1)) > 
                       (np.roll(df_12h['low'], 1) - df_12h['low']), 
                       np.maximum(df_12h['high'] - np.roll(df_12h['high'], 1), 0), 0)
    dm_minus = np.where((np.roll(df_12h['low'], 1) - df_12h['low']) > 
                        (df_12h['high'] - np.roll(df_12h['high'], 1)), 
                        np.maximum(np.roll(df_12h['low'], 1) - df_12h['low'], 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = WilderSmooth(tr.values, 14)
    dm_plus_smooth = WilderSmooth(dm_plus, 14)
    dm_minus_smooth = WilderSmooth(dm_minus, 14)
    
    # DX and ADX
    dx = np.full_like(close, np.nan)
    mask = (atr > 0) & ~np.isnan(atr) & ~np.isnan(dm_plus_smooth) & ~np.isnan(dm_minus_smooth)
    dx[mask] = 100 * np.abs(dm_plus_smooth[mask] - dm_minus_smooth[mask]) / (dm_plus_smooth[mask] + dm_minus_smooth[mask])
    adx_12h = WilderSmooth(dx, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Previous day's Camarilla levels (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    ph = df_1d['high'].shift(1).values
    pl = df_1d['low'].shift(1).values
    pc = df_1d['close'].shift(1).values
    
    rang = ph - pl
    r1 = pc + (rang * 1.1 / 12)
    s1 = pc - (rang * 1.1 / 12)
    r4 = pc + (rang * 1.1 / 2)
    s4 = pc - (rang * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # ADX filter: only trade when ADX > 25 (trending market)
        strong_trend = adx_12h_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above R1 with volume and strong trend
            if (close[i] > r1_aligned[i] and 
                volume_confirm[i] and 
                strong_trend):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and strong trend
            elif (close[i] < s1_aligned[i] and 
                  volume_confirm[i] and 
                  strong_trend):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below S1 or trend weakens (ADX < 20)
            if (close[i] < s1_aligned[i]) or (adx_12h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above R1 or trend weakens (ADX < 20)
            if (close[i] > r1_aligned[i]) or (adx_12h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals