#!/usr/bin/env python3
"""
1d_WeeklyPivot_R3S3_Breakout_WeeklyTrend
Hypothesis: Use weekly-derived R3/S3 levels as breakout triggers on daily timeframe with weekly EMA34 trend filter. 
Go long when price > weekly EMA34 and breaks above weekly R3, short when price < weekly EMA34 and breaks below weekly S3.
Exit on opposite level touch. Designed to capture weekly trends while avoiding false breakouts in ranging markets.
Weekly timeframe reduces noise and avoids overtrading, targeting 15-25 trades/year for low fee drag.
Works in bull (breakouts with trend) and bear (mean reversion at extremes with trend filter).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Calculate weekly R3/S3 levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous week's OHLC for Camarilla calculation
    prev_close = df_1w['close'].shift(1).values
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    
    # Camarilla R3 and S3 levels (weekly)
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align weekly levels to daily timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # Weekly EMA34 for trend filter
    ema_34 = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for weekly EMA
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i])):
            signals[i] = 0.0
            continue
        
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        ema_34_val = ema_34_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3 with weekly uptrend
            if close[i] > camarilla_r3_val and close[i] > ema_34_val:
                signals[i] = size
                position = 1
            # Short: price breaks below S3 with weekly downtrend
            elif close[i] < camarilla_s3_val and close[i] < ema_34_val:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 (opposite level)
            if close[i] < camarilla_s3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above R3 (opposite level)
            if close[i] > camarilla_r3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyPivot_R3S3_Breakout_WeeklyTrend"
timeframe = "1d"
leverage = 1.0