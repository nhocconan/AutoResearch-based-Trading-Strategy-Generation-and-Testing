#!/usr/bin/env python3
"""
Hypothesis: 1d strategy using weekly pivot point breakout with volume confirmation and trend filter.
- Calculate weekly pivot points (P, R1, S1, R2, S2) from previous week OHLC
- Enter long when price breaks above R2 with volume > 1.5x 20-period volume MA and price above weekly EMA20
- Enter short when price breaks below S2 with volume > 1.5x 20-period volume MA and price below weekly EMA20
- Exit when price crosses back to the opposite level (S2 for longs, R2 for shorts)
- Fixed position size 0.25 to manage drawdown
- Uses weekly trend filter to avoid counter-trend trades
- Designed for 1d timeframe with strict entry conditions to limit trades to 30-100 total over 4 years
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot point calculation and EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate weekly pivot points from previous week's OHLC
    # Pivot Point (P) = (High + Low + Close) / 3
    # Resistance 1 (R1) = (2 * P) - Low
    # Support 1 (S1) = (2 * P) - High
    # Resistance 2 (R2) = P + (High - Low)
    # Support 2 (S2) = P - (High - Low)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    P = (high_1w + low_1w + close_1w) / 3.0
    R1 = (2 * P) - low_1w
    S1 = (2 * P) - high_1w
    R2 = P + (high_1w - low_1w)
    S2 = P - (high_1w - low_1w)
    
    # Align weekly pivot points to daily timeframe (use previous week's levels)
    P_aligned = align_htf_to_ltf(prices, df_1w, P)
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
            np.isnan(R2_aligned[i]) or np.isnan(S2_aligned[i]) or 
            np.isnan(ema_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        R2_val = R2_aligned[i]
        S2_val = S2_aligned[i]
        ema_val = ema_20_aligned[i]
        
        if position == 0:
            # Look for weekly pivot level breakouts with volume confirmation and trend filter
            # Long: price breaks above R2 + volume spike + price above weekly EMA20
            if price > R2_val and vol > 1.5 * vol_ma and price > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S2 + volume spike + price below weekly EMA20
            elif price < S2_val and vol > 1.5 * vol_ma and price < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price crosses below S2 (opposite level)
            if price < S2_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price crosses above R2 (opposite level)
            if price > R2_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyPivotBreakout_Volume_EMA20"
timeframe = "1d"
leverage = 1.0