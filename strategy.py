#!/usr/bin/env python3
"""
6h_Pivots_R1_S1_Breakout_Volume_Regime_v1
6-hour strategy using daily Camarilla pivot levels (R1/S1) with volume confirmation and regime filter.
Enters long on break above R1 with volume, short on break below S1 with volume.
Exits when price returns to pivot point (PP).
Uses 12h ADX to filter ranging markets (ADX < 25).
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # === Daily Camarilla Pivot Levels ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and levels
    pp = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1 = pp + (range_1d * 1.0 / 8.0)
    s1 = pp - (range_1d * 1.0 / 8.0)
    r4 = pp + (range_1d * 1.5 / 8.0)  # Not used but calculated for reference
    s4 = pp - (range_1d * 1.5 / 8.0)
    
    # Align to 6h timeframe (wait for daily close)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Daily Volume Confirmation (20-period average) ===
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # === 12h ADX for Regime Filter (trending only) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX components
    plus_dm = np.zeros_like(high_12h)
    minus_dm = np.zeros_like(low_12h)
    plus_dm[1:] = np.maximum(high_12h[1:] - high_12h[:-1], 0)
    minus_dm[1:] = np.maximum(low_12h[:-1] - low_12h[1:], 0)
    plus_dm = np.where(plus_dm > minus_dm, plus_dm, 0)
    minus_dm = np.where(minus_dm > plus_dm, minus_dm, 0)
    
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = high_12h[0] - low_12h[0]  # First TR
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / (atr * 14)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / (atr * 14)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current day's volume for confirmation
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        vol_confirmed = vol_1d_current > 1.5 * vol_ma_1d_aligned[i]
        
        # Regime filter: only trade in trending markets (ADX > 25)
        trending = adx_aligned[i] > 25
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above R1 with volume confirmation and trending
            if (close[i] > r1_aligned[i] and vol_confirmed and trending):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below S1 with volume confirmation and trending
            elif (close[i] < s1_aligned[i] and vol_confirmed and trending):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: price returns to pivot point
        elif position == 1:
            # Exit long: price crosses below pivot point
            if close[i] < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above pivot point
            if close[i] > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Pivots_R1_S1_Breakout_Volume_Regime_v1"
timeframe = "6h"
leverage = 1.0