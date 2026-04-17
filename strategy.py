#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using 1-week pivot levels (R1/S1, R2/S2) with volume confirmation and 1-day EMA trend filter.
- Weekly pivot levels calculated from prior week OHLC
- Enter long when price breaks above R1 with volume > 1.5x 20-period volume MA and price above 1d EMA50
- Enter short when price breaks below S1 with volume > 1.5x 20-period volume MA and price below 1d EMA50
- Exit when price crosses back to opposite pivot level (S1 for longs, R1 for shorts)
- Fixed position size 0.25 to manage drawdown
- Designed for 6h timeframe with strict entry conditions to limit trades to 50-150 total over 4 years
- Uses weekly structure for trend context and daily EMA for intermediate trend filter
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
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Get daily data for EMA filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate weekly pivot points from previous week's OHLC
    # Standard pivot point: P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    # R2 = P + (H - L)
    # S2 = P - (H - L)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot = (high_1w + low_1w + close_1w) / 3.0
    R1 = 2 * pivot - low_1w
    S1 = 2 * pivot - high_1w
    R2 = pivot + (high_1w - low_1w)
    S2 = pivot - (high_1w - low_1w)
    
    # Align weekly pivot levels to 6h timeframe (use previous week's levels)
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    R2_aligned = align_htf_to_ltf(prices, df_1w, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1w, S2)
    
    # Volume confirmation: 20-period volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(volume_ma_20.iloc[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(R2_aligned[i]) or np.isnan(S2_aligned[i]) or
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        R1_val = R1_aligned[i]
        S1_val = S1_aligned[i]
        R2_val = R2_aligned[i]
        S2_val = S2_aligned[i]
        ema_val = ema_50_aligned[i]
        
        if position == 0:
            # Look for weekly pivot level breakouts with volume confirmation and trend filter
            # Long: price breaks above R1 + volume spike + price above 1d EMA50
            if price > R1_val and vol > 1.5 * vol_ma and price > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume spike + price below 1d EMA50
            elif price < S1_val and vol > 1.5 * vol_ma and price < ema_val:
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

name = "6h_WeeklyPivot_R1S1_Volume_1dEMA50"
timeframe = "6h"
leverage = 1.0