#!/usr/bin/env python3
"""
Hypothesis: 12h strategy using 1-week EMA50 trend filter with 1-day pivot point breakout and volume confirmation.
- Uses weekly EMA50 for long-term trend direction (avoids counter-trend trades)
- Enters long when price breaks above daily R1 with volume > 1.8x 20-period volume MA and price above weekly EMA50
- Enters short when price breaks below daily S1 with volume > 1.8x 20-period volume MA and price below weekly EMA50
- Exits when price crosses back to the opposite pivot level (S1 for longs, R1 for shorts)
- Fixed position size 0.25 to manage drawdown
- Designed for 12h timeframe with strict entry conditions to limit trades to 50-150 total over 4 years
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
    
    # Get 1-day data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1-week data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily pivot points from previous day's OHLC
    # Pivot Point = (High + Low + Close) / 3
    # R1 = 2*P - Low
    # S1 = 2*P - High
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    R1 = 2 * pivot - low_1d
    S1 = 2 * pivot - high_1d
    
    # Align pivot levels to 12h timeframe (use previous day's levels)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation: 20-period volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(volume_ma_20.iloc[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        R1_val = R1_aligned[i]
        S1_val = S1_aligned[i]
        ema_val = ema_50_1w_aligned[i]
        
        if position == 0:
            # Look for pivot point breakouts with volume confirmation and trend filter
            # Long: price breaks above R1 + volume spike + price above weekly EMA50
            if price > R1_val and vol > 1.8 * vol_ma and price > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume spike + price below weekly EMA50
            elif price < S1_val and vol > 1.8 * vol_ma and price < ema_val:
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

name = "12h_PivotPointBreakout_Volume_WeeklyEMA50"
timeframe = "12h"
leverage = 1.0