#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyPivot_R1S1_Breakout_Volume_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot levels
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    r1_1w = close_1w + range_1w * 1.1 / 12.0
    s1_1w = close_1w - range_1w * 1.1 / 12.0
    
    # Align weekly pivot levels to daily timeframe
    pivot_1d = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1d = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1d = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Daily ADX for trend strength (14-period)
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
    
    di_plus = 100 * dm_plus14 / tr14
    di_minus = 100 * dm_minus14 / tr14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(pivot_1d[i]) or np.isnan(r1_1d[i]) or np.isnan(s1_1d[i]) or \
           np.isnan(adx[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        adx_val = adx[i]
        pivot = pivot_1d[i]
        r1 = r1_1d[i]
        s1 = s1_1d[i]
        
        volume_confirmed = vol > 2.0 * vol_ma
        trending = adx_val > 25  # Strong trend filter
        
        if position == 0:
            # Long: Price breaks above R1 + volume + trending
            if price > r1 and volume_confirmed and trending:
                signals[i] = 0.30
                position = 1
            # Short: Price breaks below S1 + volume + trending
            elif price < s1 and volume_confirmed and trending:
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Exit: Price returns below pivot
            if price < pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit: Price returns above pivot
            if price > pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals