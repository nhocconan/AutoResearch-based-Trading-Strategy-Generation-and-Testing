#!/usr/bin/env python3
"""
Hypothesis: 1d strategy using weekly pivot levels with volume confirmation and ADX trend filter.
- Calculate weekly pivot points (R1, S1, R2, S2) from previous week's OHLC
- Enter long when price breaks above R1 with volume > 1.5x 20-period volume MA and ADX > 25 (trending)
- Enter short when price breaks below S1 with volume > 1.5x 20-period volume MA and ADX > 25 (trending)
- Exit when price crosses back to the opposite pivot level (S1 for longs, R1 for shorts)
- Fixed position size 0.25 to manage drawdown
- Uses weekly trend filter to avoid counter-trend trades
- Designed for 1d timeframe with strict entry conditions to limit trades to 30-100 total over 4 years
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
    
    # Get weekly data for pivot calculation and ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate ADX(14) for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate weekly pivot points from previous week's OHLC
    # Pivot Point (P) = (High + Low + Close) / 3
    # R1 = 2*P - Low
    # S1 = 2*P - High
    # R2 = P + (High - Low)
    # S2 = P - (High - Low)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    P = (high_1w + low_1w + close_1w) / 3.0
    R1 = 2 * P - low_1w
    S1 = 2 * P - high_1w
    R2 = P + (high_1w - low_1w)
    S2 = P - (high_1w - low_1w)
    
    # Align weekly data to daily timeframe (use previous week's levels)
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    R2_aligned = align_htf_to_ltf(prices, df_1w, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1w, S2)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume confirmation: 20-period volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(volume_ma_20.iloc[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(R2_aligned[i]) or np.isnan(S2_aligned[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        R1_val = R1_aligned[i]
        S1_val = S1_aligned[i]
        R2_val = R2_aligned[i]
        S2_val = S2_aligned[i]
        adx_val = adx_aligned[i]
        
        if position == 0:
            # Look for pivot level breakouts with volume confirmation and trend filter
            # Long: price breaks above R1 + volume spike + ADX > 25 (trending)
            if price > R1_val and vol > 1.5 * vol_ma and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume spike + ADX > 25 (trending)
            elif price < S1_val and vol > 1.5 * vol_ma and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price crosses below S1 (opposite level)
            if price < S1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price crosses above R1 (opposite level)
            if price > R1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyPivot_R1S1_Volume_ADX25"
timeframe = "1d"
leverage = 1.0