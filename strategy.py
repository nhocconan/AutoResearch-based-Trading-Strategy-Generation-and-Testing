#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 1-day pivot point breakouts with volume confirmation and 1-week EMA trend filter.
- Calculate daily pivot points (P), support (S1, S2) and resistance (R1, R2) from prior day's OHLC
- Enter long when price breaks above R1 with volume > 1.5x 20-period volume MA and price above 1-week EMA50
- Enter short when price breaks below S1 with volume > 1.5x 20-period volume MA and price below 1-week EMA50
- Exit when price returns to pivot point (P)
- Fixed position size 0.25 to manage drawdown
- Uses institutional pivot levels that work in both accumulation and distribution phases
- Designed for 4h timeframe with strict entry conditions to limit trades to 75-200 total over 4 years
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
    
    # Get 1-day data for pivot points (calculate from prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points: P = (H+L+C)/3, S1 = 2P - H, S2 = P - (H-L), R1 = 2P - L, R2 = P + (H-L)
    # We need the prior day's data, so shift by 1
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot points for each day (using prior day's data)
    pivot = (np.roll(high_1d, 1) + np.roll(low_1d, 1) + np.roll(close_1d, 1)) / 3
    s1 = 2 * pivot - np.roll(high_1d, 1)
    s2 = pivot - (np.roll(high_1d, 1) - np.roll(low_1d, 1))
    r1 = 2 * pivot - np.roll(low_1d, 1)
    r2 = pivot + (np.roll(high_1d, 1) - np.roll(low_1d, 1))
    
    # Align pivot levels to 4h timeframe (wait for daily bar to close)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    
    # Volume confirmation: 20-period volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    # Trend filter: 1-week EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        ema_50_1w = np.full(len(close), np.nan)
    else:
        ema_50_1w_raw = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean()
        ema_50_1w = align_htf_to_ltf(prices, df_1w, ema_50_1w_raw.values)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(volume_ma_20.iloc[i]) or 
            np.isnan(ema_50_1w[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        pivot_val = pivot_aligned[i]
        s1_val = s1_aligned[i]
        r1_val = r1_aligned[i]
        ema_val = ema_50_1w[i]
        
        if position == 0:
            # Look for pivot breakouts with volume confirmation and trend filter
            # Long: price breaks above R1 + volume spike + price above 1-week EMA50
            if price > r1_val and vol > 1.5 * vol_ma and price > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume spike + price below 1-week EMA50
            elif price < s1_val and vol > 1.5 * vol_ma and price < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price returns to pivot point (mean reversion to fair value)
            if price >= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price returns to pivot point (mean reversion to fair value)
            if price <= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_PivotPointBreakout_Volume_1wEMA50"
timeframe = "4h"
leverage = 1.0