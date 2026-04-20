#!/usr/bin/env python3
"""
1h_4h_Structure_Entry_With_1d_Trend - 
Hypothesis: In 1h timeframe, use 4h structure (HH/HL or LH/LL) as direction filter,
combined with 1d EMA trend filter for long-term bias, and enter on pullbacks to 4h EMA.
This reduces overtrading by requiring multiple timeframe alignment while keeping
trades frequent enough (target 15-37/year) for statistical significance.
Works in bull/bear via 1d trend filter - only trade in direction of higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # === 1D TREND FILTER (EMA34) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 4H STRUCTURE AND EMA ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # 4h EMA21 for dynamic pullback entries
    ema_21_4h = pd.Series(df_4h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # 4h structure: Higher Highs/Higher Lows for uptrend, Lower Highs/Lower Lows for downtrend
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Track 4h swing points
    hh_4h = np.full_like(high_4h, np.nan)  # Higher Highs
    hl_4h = np.full_like(low_4h, np.nan)   # Higher Lows
    lh_4h = np.full_like(high_4h, np.nan)  # Lower Highs
    ll_4h = np.full_like(low_4h, np.nan)   # Lower Lows
    
    # Simple swing detection: look for pivot points
    for i in range(2, len(high_4h)):
        # Higher High: current high > previous high and previous high > one before
        if high_4h[i] > high_4h[i-1] and high_4h[i-1] > high_4h[i-2]:
            hh_4h[i-1] = high_4h[i-1]
        # Higher Low: current low > previous low and previous low > one before  
        if low_4h[i] > low_4h[i-1] and low_4h[i-1] > low_4h[i-2]:
            hl_4h[i-1] = low_4h[i-1]
        # Lower High: current high < previous high and previous high < one before
        if high_4h[i] < high_4h[i-1] and high_4h[i-1] < high_4h[i-2]:
            lh_4h[i-1] = high_4h[i-1]
        # Lower Low: current low < previous low and previous low < one before
        if low_4h[i] < low_4h[i-1] and low_4h[i-1] < low_4h[i-2]:
            ll_4h[i-1] = low_4h[i-1]
    
    # Align 4h structure to 1h
    hh_4h_aligned = align_htf_to_ltf(prices, df_4h, hh_4h)
    hl_4h_aligned = align_htf_to_ltf(prices, df_4h, hl_4h)
    lh_4h_aligned = align_htf_to_ltf(prices, df_4h, lh_4h)
    ll_4h_aligned = align_htf_to_ltf(prices, df_4h, ll_4h)
    
    # === 1H DATA ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if critical values are NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_21_4h_aligned[i]) or
            np.isnan(hh_4h_aligned[i]) or np.isnan(hl_4h_aligned[i]) or
            np.isnan(lh_4h_aligned[i]) or np.isnan(ll_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_1d = ema_34_1d_aligned[i]
        ema_4h = ema_21_4h_aligned[i]
        
        # Determine 4h trend structure (need recent swing points)
        lookback = min(50, i//4)  # Look back ~200 hours max
        start_idx = max(0, i - lookback)
        
        # Check for recent HH/HL (uptrend structure) or LH/LL (downtrend structure)
        recent_hh = hh_4h_aligned[start_idx:i+1]
        recent_hl = hl_4h_aligned[start_idx:i+1]
        recent_lh = lh_4h_aligned[start_idx:i+1]
        recent_ll = ll_4h_aligned[start_idx:i+1]
        
        # Valid if we have at least one of each structure type
        uptrend_structure = (np.nansum(~np.isnan(recent_hh)) >= 1 and 
                            np.nansum(~np.isnan(recent_hl)) >= 1)
        downtrend_structure = (np.nansum(~np.isnan(recent_lh)) >= 1 and 
                              np.nansum(~np.isnan(recent_ll)) >= 1)
        
        # Volume filter: current volume > 1.5x 20-period average
        vol_ma = np.mean(volume[max(0, i-20):i]) if i >= 20 else volume[i]
        vol_ok = volume[i] > 1.5 * vol_ma
        
        if position == 0:
            # LONG: 1d uptrend + 4h uptrend structure + pullback to 4h EMA
            if (price > ema_1d and uptrend_structure and 
                price <= ema_4h * 1.005 and price >= ema_4h * 0.995 and vol_ok):
                signals[i] = 0.20
                position = 1
            # SHORT: 1d downtrend + 4h downtrend structure + pullback to 4h EMA
            elif (price < ema_1d and downtrend_structure and
                  price <= ema_4h * 1.005 and price >= ema_4h * 0.995 and vol_ok):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # LONG EXIT: 1d trend breaks or 4h structure fails or break below 4h EMA
            if (price < ema_1d or not uptrend_structure or 
                price < ema_4h * 0.98):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # SHORT EXIT: 1d trend breaks or 4h structure fails or break above 4h EMA
            if (price > ema_1d or not downtrend_structure or 
                price > ema_4h * 1.02):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h_Structure_Entry_With_1d_Trend"
timeframe = "1h"
leverage = 1.0