#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 reversal with 1w EMA200 trend filter and volume confirmation.
Long when price touches S3 support AND price > 1w EMA200 AND volume > 1.3x average.
Short when price touches R3 resistance AND price < 1w EMA200 AND volume > 1.3x average.
Exit on opposite Camarilla level touch (R3 for longs, S3 for shorts) or volume drop.
Camarilla levels provide precise intraday support/resistance from prior 1d range.
1w EMA200 ensures trading with primary weekly trend, avoiding counter-trend whipsaws.
Volume confirmation ensures institutional participation at key levels.
Designed for 6h timeframe targeting 50-150 total trades over 4 years with selective entries.
Works in bull markets (buying dips in uptrend) and bear markets (selling rallies in downtrend).
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
    
    # Load 1d data for Camarilla calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Load 1w data for EMA200 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA200 on 1w data
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1w EMA200 to 6h timeframe
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if 1w EMA200 not ready
        if np.isnan(ema200_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Need prior 1d bar for Camarilla calculation (yesterday's HLC)
        if i < 96:  # Need at least 4x 6h bars back for 1d bar (6h*4=24h)
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get prior completed 1d bar (yesterday)
        idx_1d = i // 4  # 4x 6h bars = 1d bar
        if idx_1d >= len(df_1d):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Prior 1d bar data
        high_1d = df_1d['high'].iloc[idx_1d - 1]
        low_1d = df_1d['low'].iloc[idx_1d - 1]
        close_1d = df_1d['close'].iloc[idx_1d - 1]
        
        # Calculate Camarilla levels for today
        range_1d = high_1d - low_1d
        if range_1d <= 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        camarilla_mult = range_1d / 12.0
        r3 = close_1d + camarilla_mult * 1.1
        s3 = close_1d - camarilla_mult * 1.1
        
        # Volume average (20-period) on primary timeframe
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        if np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        price = close[i]
        vol_current = volume[i]
        ema200_val = ema200_1w_aligned[i]
        
        if position == 0:
            # Long: Price touches S3 support AND above 1w EMA200 AND volume spike
            if (price <= s3 and price > ema200_val and vol_current > 1.3 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price touches R3 resistance AND below 1w EMA200 AND volume spike
            elif (price >= r3 and price < ema200_val and vol_current > 1.3 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price touches R3 resistance OR volume drops below average
                if (price >= r3 or vol_current < vol_ma[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price touches S3 support OR volume drops below average
                if (price <= s3 or vol_current < vol_ma[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Camarilla_R3S3_1wEMA200_Volume"
timeframe = "6h"
leverage = 1.0