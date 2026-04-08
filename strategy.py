#!/usr/bin/env python3
# 12h Daily Pivot Breakout with Volume Confirmation and Choppiness Regime
# Hypothesis: Daily pivot points act as key institutional support/resistance.
# Breakouts above R1 or below S1 with volume > 1.3x 20-period average and in trending markets (Choppiness Index < 40) indicate institutional participation.
# Works in bull/bear markets by capturing breakouts from key levels with regime filter to avoid chop.
# Target: 15-35 trades/year per symbol.

name = "12h_daily_pivot_breakout_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot points and volume filter - call ONCE before loop
    df_d = get_htf_data(prices, '1d')
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    volume_d = df_d['volume'].values
    
    # Calculate daily pivot points (standard floor trader pivots)
    pp_d = (high_d + low_d + close_d) / 3
    r1_d = 2 * pp_d - low_d  # R1 = (2*P) - L
    s1_d = 2 * pp_d - high_d # S1 = (2*P) - H
    
    # Calculate 20-period average volume for daily timeframe
    vol_ma_d = pd.Series(volume_d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index on daily timeframe (trending vs ranging filter)
    # True Range
    tr1 = high_d - low_d
    tr2 = np.abs(high_d - np.roll(close_d, 1))
    tr3 = np.abs(low_d - np.roll(close_d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Sum of True Range over 14 periods
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh14 = pd.Series(high_d).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low_d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(ATR14 / (HH14 - LL14)) / log10(14)
    # Add small epsilon to avoid division by zero
    range14 = hh14 - ll14
    range14 = np.where(range14 == 0, 1e-10, range14)
    chop = 100 * np.log10(atr14 / range14) / np.log10(14)
    # Handle any remaining invalid values
    chop = np.where(np.isnan(chop) | np.isinf(chop), 50, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 30
    
    for i in range(start_idx, n):
        # Get aligned daily values for current 12h bar
        r1 = align_htf_to_ltf(prices, df_d, r1_d)[i]
        s1 = align_htf_to_ltf(prices, df_d, s1_d)[i]
        vol_ma = align_htf_to_ltf(prices, df_d, vol_ma_d)[i]
        chop_val = align_htf_to_ltf(prices, df_d, chop)[i]
        
        # Skip if any required data is NaN
        if np.isnan(r1) or np.isnan(s1) or np.isnan(vol_ma) or np.isnan(chop_val) or volume[i] == 0:
            signals[i] = 0.0
            continue
        
        # Volume breakout condition: current volume > 1.3x 20-period average
        vol_breakout = volume[i] > 1.3 * vol_ma
        
        # Trending market condition: Choppiness Index < 40 (lower = more trending)
        trending_market = chop_val < 40
        
        if position == 1:  # Long position
            # Exit if price breaks below S1 (failed breakout)
            if close[i] < s1:
                position = 0
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price breaks above R1 (failed breakout)
            if close[i] > r1:
                position = 0
                signals[i] = 0.0
            elif position == -1:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long above R1 with volume confirmation and trending market
            if close[i] > r1 and vol_breakout and trending_market:
                position = 1
                signals[i] = 0.25
            # Breakout short below S1 with volume confirmation and trending market
            elif close[i] < s1 and vol_breakout and trending_market:
                position = -1
                signals[i] = -0.25
    
    return signals