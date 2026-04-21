#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_With_Volume_Spike
Hypothesis: Camarilla pivot levels from 1d timeframe provide strong support/resistance.
Buy when price breaks above R1 with volume spike, sell when breaks below S1.
Use 1d ADX > 25 to filter for trending markets only. Designed for low trade frequency
(12-37 trades/year) to minimize fee flood while capturing trends in both bull and bear.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Camarilla pivot levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1 = close_1d + (range_1d * 1.1 / 12)
    s1 = close_1d - (range_1d * 1.1 / 12)
    
    # === 1d ADX(14) for trend filter ===
    # Calculate directional movement
    high_diff = np.diff(high_1d)
    low_diff = -np.diff(low_1d)  # inverted for calculation
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # True range
    tr1 = np.abs(np.diff(high_1d))
    tr2 = np.abs(np.diff(low_1d))
    tr3 = np.abs(np.diff(close_1d))
    tr = np.maximum.reduce([tr1, tr2, tr3])
    
    # Add first element (no diff)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(arr[1:period]) if period > 1 else arr[0]
        # Wilder smoothing: today = (yesterday * (period-1) + today) / period
        for i in range(period, len(arr)):
            if np.isnan(result[i-1]) or np.isnan(arr[i]):
                result[i] = np.nan
            else:
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    period = 14
    atr = wilder_smooth(tr, period)
    plus_di = 100 * wilder_smooth(plus_dm, period) / atr
    minus_di = 100 * wilder_smooth(minus_dm, period) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, period)
    
    # === 1d Volume spike detection ===
    vol_ma = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = df_1d['volume'].values / vol_ma
    
    # Align all 1d indicators to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        adx_val = adx_aligned[i]
        vol_ratio_val = vol_ratio_aligned[i]
        
        if position == 0:
            # Long: Break above R1 with volume spike and trending market (ADX > 25)
            if (price_close > r1_val and 
                vol_ratio_val > 1.5 and 
                adx_val > 25):
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with volume spike and trending market (ADX > 25)
            elif (price_close < s1_val and 
                  vol_ratio_val > 1.5 and 
                  adx_val > 25):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price returns to pivot level (mean reversion within day)
            # Calculate daily pivot for exit
            daily_pivot = (high_1d[i//12] + low_1d[i//12] + close_1d[i//12]) / 3.0 if i//12 < len(close_1d) else pivot[-1]
            daily_pivot_aligned = align_htf_to_ltf(prices, df_1d, np.full_like(close_1d, daily_pivot))[i] if i//12 < len(close_1d) else pivot[-1]
            
            if position == 1 and price_close < daily_pivot_aligned:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > daily_pivot_aligned:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_With_Volume_Spike"
timeframe = "12h"
leverage = 1.0