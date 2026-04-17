#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using weekly pivot points (R2/S2) with volume confirmation and 1-day trend filter.
- Enter long when price breaks above R2 with volume > 1.5x 20-period volume MA and price > 1-day EMA200
- Enter short when price breaks below S2 with volume > 1.5x 20-period volume MA and price < 1-day EMA200
- Exit when price crosses back below R2 (long) or above S2 (short)
- Weekly pivot points calculated from weekly high/low/close
- Weekly data is considered as the higher timeframe (HTF) for pivot calculation
- Designed for 6h timeframe with strict entry conditions to limit trades to 50-150 total over 4 years
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
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3, R2 = P + (H-L), S2 = P - (H-L)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    
    # Align weekly pivot levels to 6h timeframe
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    
    # Get 1-day data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA200 for trend filter
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume confirmation: 20-period volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # warmup for EMA200
    
    for i in range(start_idx, n):
        if (np.isnan(volume_ma_20.iloc[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(ema_200_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        r2_level = r2_aligned[i]
        s2_level = s2_aligned[i]
        ema_val = ema_200_aligned[i]
        
        if position == 0:
            # Look for breakout with volume confirmation and trend filter
            # Long: price breaks above R2, volume spike, price above EMA200
            if price > r2_level and vol > 1.5 * vol_ma and price > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S2, volume spike, price below EMA200
            elif price < s2_level and vol > 1.5 * vol_ma and price < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price crosses back below R2
            if price < r2_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price crosses back above S2
            if price > s2_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R2S2_Volume_1dEMA200"
timeframe = "6h"
leverage = 1.0