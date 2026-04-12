#!/usr/bin/env python3
"""
4h_1d_Structure_Breakout_v1
Hypothesis: Daily structure (higher highs/lows) defines trend; 4h Donchian breakout enters in trend direction with volume confirmation. Works in bull/bear by following higher timeframe structure. Low trade frequency via structure filter and volume spike requirement.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Structure_Breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for structure and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily structure: Higher Highs and Higher Lows (uptrend) or Lower Highs and Lower Lows (downtrend)
    # Use 20-period lookback for structure
    dh = df_1d['high'].rolling(window=20, min_periods=20).max().values  # Daily Higher High
    dl = df_1d['low'].rolling(window=20, min_periods=20).min().values   # Daily Lower Low
    hh = df_1d['high'].rolling(window=20, min_periods=20).max().values  # Daily Highest High (same as dh, but conceptually)
    hl = df_1d['low'].rolling(window=20, min_periods=20).min().values   # Daily Lowest Low (same as dl)
    
    # Uptrend: Today's close > yesterday's HH and today's low > yesterday's HL
    # Downtrend: Today's close < yesterday's LH and today's high < yesterday's HL
    # Simplified: Higher Highs/Lows structure
    dh_prev = np.roll(dh, 1)  # Previous day's Higher High
    dl_prev = np.roll(dl, 1)  # Previous day's Lower Low
    hh_prev = np.roll(hh, 1)  # Previous day's Highest High
    hl_prev = np.roll(hl, 1)  # Previous day's Lowest Low
    
    # Handle first value
    dh_prev[0] = dh[0]
    dl_prev[0] = dl[0]
    hh_prev[0] = hh[0]
    hl_prev[0] = hl[0]
    
    # Structure signals: 1 = uptrend structure, -1 = downtrend structure, 0 = unclear
    struct_up = (df_1d['close'].values > hh_prev) & (df_1d['low'].values > hl_prev)
    struct_down = (df_1d['close'].values < ll_prev) & (df_1d['high'].values < lh_prev) if False else (df_1d['close'].values < dh_prev) & (df_1d['high'].values < hh_prev)
    # Fix: Actually need to compute Lower High and Higher Low properly
    lh = df_1d['high'].rolling(window=20, min_periods=20).max().values  # Same as HH for simplicity in this context
    hl_val = df_1d['low'].rolling(window=20, min_periods=20).min().values # Same as LL
    
    # Proper structure: Higher High and Higher Low = uptrend
    hh_20 = df_1d['high'].rolling(window=20, min_periods=20).max().values
    hl_20 = df_1d['low'].rolling(window=20, min_periods=20).min().values
    lh_20 = df_1d['high'].rolling(window=20, min_periods=20).max().values  # Simplified
    ll_20 = df_1d['low'].rolling(window=20, min_periods=20).min().values   # Simplified
    
    # Shift to get previous values
    hh_20_prev = np.roll(hh_20, 1)
    hl_20_prev = np.roll(hl_20, 1)
    hh_20_prev[0] = hh_20[0]
    hl_20_prev[0] = hl_20[0]
    
    # Uptrend: Higher High (close > prev HH) and Higher Low (low > prev HL)
    struct_up = (df_1d['close'].values > hh_20_prev) & (df_1d['low'].values > hl_20_prev)
    # Downtrend: Lower High (high < prev LH) and Lower Low (low < prev LL) - simplified
    lh_20 = df_1d['high'].rolling(window=20, min_periods=20).max().values
    ll_20 = df_1d['low'].rolling(window=20, min_periods=20).min().values
    lh_20_prev = np.roll(lh_20, 1)
    ll_20_prev = np.roll(ll_20, 1)
    lh_20_prev[0] = lh_20[0]
    ll_20_prev[0] = ll_20[0]
    struct_down = (df_1d['high'].values < lh_20_prev) & (df_1d['low'].values < ll_20_prev)
    
    # Structure signal: 1 for uptrend, -1 for downtrend, 0 otherwise
    struct_signal = np.where(struct_up, 1, np.where(struct_down, -1, 0))
    
    # Align structure to 4h
    struct_aligned = align_htf_to_ltf(prices, df_1d, struct_signal)
    
    # Daily volume average (20-period) for volume spike
    vol_ma = df_1d['volume'].rolling(window=20, min_periods=20).mean().values
    vol_ratio = df_1d['volume'].values / vol_ma
    vol_ratio = np.where(vol_ma > 0, vol_ratio, 1.0)
    vol_ratio_prev = np.roll(vol_ratio, 1)
    vol_ratio_prev[0] = vol_ratio[0]
    vol_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_prev)
    
    # 4h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        # Skip if any required data is invalid
        if (np.isnan(struct_aligned[i]) or np.isnan(vol_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Structure filter: only trade in direction of daily structure
        struct_bull = struct_aligned[i] > 0
        struct_bear = struct_aligned[i] < 0
        
        # Volume confirmation: volume spike > 1.5x average
        vol_spike = vol_aligned[i] > 1.5
        
        # Donchian breakout
        breakout_up = high[i] > highest_high[i-1] if i > 0 else False
        breakout_down = low[i] < lowest_low[i-1] if i > 0 else False
        
        # Entry conditions: breakout in direction of structure with volume
        long_entry = struct_bull and breakout_up and vol_spike
        short_entry = struct_bear and breakout_down and vol_spike
        
        # Exit conditions: opposite breakout or structure change
        long_exit = breakout_down or (struct_aligned[i] < 0)
        short_exit = breakout_up or (struct_aligned[i] > 0)
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals