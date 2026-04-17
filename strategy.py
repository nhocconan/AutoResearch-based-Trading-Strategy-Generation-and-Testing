#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using weekly pivot R1/S1 levels with volume confirmation and 1d EMA50 trend filter.
- Long when price breaks above weekly pivot R1 + volume > 1.5x 20-period 6h volume MA + price above 1d EMA50
- Short when price breaks below weekly pivot S1 + volume > 1.5x 20-period 6h volume MA + price below 1d EMA50
- Fixed position size 0.25 to manage drawdown in bear markets
- Uses proven edge: weekly pivot levels (structure) + volume spike + HTF trend
- Designed for 6h timeframe with strict entry conditions to limit trades to 50-150 total over 4 years
- Works in bull markets (buying breakouts with uptrend) and bear markets (selling breakdowns with downtrend)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points (HTF for structure)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    
    # Get 1d data for EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period) on 6h for confirmation
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to primary timeframe (6h)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        ema_50_val = ema_50_aligned[i]
        vol_ma = volume_ma_20[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and 1d EMA50 trend filter
            # Long: price breaks above weekly R1 + volume spike + price above 1d EMA50
            if price > r1 and vol > 1.5 * vol_ma and price > ema_50_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below weekly S1 + volume spike + price below 1d EMA50
            elif price < s1 and vol > 1.5 * vol_ma and price < ema_50_val:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit on close below 1d EMA50 (trend change) or opposite weekly level
            if price < ema_50_val or price < s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit on close above 1d EMA50 (trend change) or opposite weekly level
            if price > ema_50_val or price > r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R1S1_VolumeSpike_1dEMA50"
timeframe = "6h"
leverage = 1.0