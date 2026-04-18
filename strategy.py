#!/usr/bin/env python3
"""
4h 1D Pivot Point Breakout with Volume Confirmation
Hypothesis: Price breaking above/below daily pivot levels (R1, S1) with volume confirmation
captures breakout moves in both bull and bear markets. Uses 1d R1/S1 pivot levels
from previous day, volume > 1.5x 20-period average, and ADX > 20 for trend strength.
Target: 20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot calculation (previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points from previous day's OHLC
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    prev_high = df_1d['high'].shift(1).values  # Previous day's high
    prev_low = df_1d['low'].shift(1).values    # Previous day's low
    prev_close = df_1d['close'].shift(1).values # Previous day's close
    
    # Calculate pivot levels for each 1d bar
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    
    # Align pivot levels to 4h timeframe (use previous day's levels)
    pivot_aligned = align_ltf_to_htf(prices, df_1d, pivot)
    r1_aligned = align_ltf_to_htf(prices, df_1d, r1)
    s1_aligned = align_ltf_to_htf(prices, df_1d, s1)
    
    # ADX for trend strength (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    di_plus = np.where(tr14 > 0, 100 * dm_plus14 / tr14, 0)
    di_minus = np.where(tr14 > 0, 100 * dm_minus14 / tr14, 0)
    
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(adx[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        adx_val = adx[i]
        vol_conf = vol_ratio[i] > 1.5
        
        if position == 0:
            # Breakout above R1 with volume and trend strength
            if adx_val > 20 and price > r1_val and vol_conf:
                signals[i] = 0.25
                position = 1
            # Breakdown below S1 with volume and trend strength
            elif adx_val > 20 and price < s1_val and vol_conf:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if price returns below pivot or trend weakens
            if price < pivot_val or adx_val < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns above pivot or trend weakens
            if price > pivot_val or adx_val < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1D_Pivot_R1S1_Breakout_Volume"
timeframe = "4h"
leverage = 1.0