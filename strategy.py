#!/usr/bin/env python3
"""
12h_1D_R1_S1_Pivot_Breakout_Volume_Conservative_v1
Concept: 12h breakout above R1 or below S1 pivots from prior 1-day, with volume confirmation and choppiness filter.
- Pivots calculated from 1D OHLC: R1 = 2*P - L, S1 = 2*P - H where P = (H+L+C)/3
- Long: Close > R1 AND volume > 1.5x avg volume AND choppiness < 61.8 (trending)
- Short: Close < S1 AND volume > 1.5x avg volume AND choppiness < 61.8 (trending)
- Exit: Close crosses back below R1 (long) or above S1 (short)
- Position sizing: 0.25
- Target: 15-35 trades/year (60-140 total over 4 years)
- Works in bull/bear: Pivot levels act as dynamic support/resistance, volume confirms breakout strength, choppiness filter avoids ranging markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1D_R1_S1_Pivot_Breakout_Volume_Conservative_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 10:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # === Calculate 1D pivot points (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # Align pivots to 12h timeframe (wait for daily close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 12h: Volume and price data ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Average volume (50-period for stability)
    avg_volume = pd.Series(volume).rolling(window=50, min_periods=20).mean().values
    
    # === 12h: Choppiness Index (trending vs ranging filter) ===
    # Using 14-period choppy index: higher = more ranging, lower = more trending
    high = prices['high'].values
    low = prices['low'].values
    
    # True Range
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # ATR (14-period)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of True Range over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(tr_sum / (atr * 14)) / log10(14)
    # Avoid division by zero and log of zero
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(tr_sum / (atr * 14)) / np.log10(14)
        chop = np.where((atr * 14) > 0, chop, 50)  # default to middle when ATR=0
        chop = np.where(tr_sum > 0, chop, 50)
    
    # Chop > 61.8 = ranging, Chop < 38.2 = trending (we want trending markets)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for volume average and chop
    
    for i in range(start_idx, n):
        # Get values
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        close_val = close[i]
        volume_val = volume[i]
        avg_vol = avg_volume[i]
        chop_val = chop[i]
        
        # Skip if any value is NaN
        if (np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(close_val) or 
            np.isnan(volume_val) or np.isnan(avg_vol) or np.isnan(chop_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = volume_val > 1.5 * avg_vol
        
        # Trending market: choppiness < 61.8 (lower = more trending)
        trending = chop_val < 61.8
        
        if position == 0:
            # Long: Close above R1 AND volume confirmed AND trending market
            if close_val > r1_val and volume_confirmed and trending:
                signals[i] = 0.25
                position = 1
            # Short: Close below S1 AND volume confirmed AND trending market
            elif close_val < s1_val and volume_confirmed and trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Close crosses back below R1
            if close_val < r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Close crosses back above S1
            if close_val > s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals