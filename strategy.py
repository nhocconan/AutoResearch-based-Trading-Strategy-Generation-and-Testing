#!/usr/bin/env python3
"""
12h_camarilla_pivot_1w_trend_volume_v1
Hypothesis: On 12h timeframe, use weekly Camarilla pivot levels (derived from 1d OHLC) as support/resistance with trend filter from 1w EMA20 and volume confirmation. Go long when price bounces off S3/S4 with bullish weekly trend and above-average volume. Go short when price rejects R3/R4 with bearish weekly trend and above-average volume. Uses weekly timeframe for trend to avoid whipsaws and focuses on institutional pivot levels that work in both bull and bear markets. Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1w_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate Camarilla pivot levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shift by 1)
    ph = np.roll(high_1d, 1)
    pl = np.roll(low_1d, 1)
    pc = np.roll(close_1d, 1)
    ph[0] = high_1d[0]  # First day uses same day
    pl[0] = low_1d[0]
    pc[0] = close_1d[0]
    
    # Camarilla levels
    range_ = ph - pl
    # Resistance levels
    r3 = pc + range_ * 1.1 / 6
    r4 = pc + range_ * 1.1 / 2
    # Support levels
    s3 = pc - range_ * 1.1 / 6
    s4 = pc - range_ * 1.1 / 2
    
    # Align to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_20_1w_aligned[i]
        weekly_downtrend = close[i] < ema_20_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price moves below S3 or trend changes
            if close[i] < s3_aligned[i] or not weekly_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves above R3 or trend changes
            if close[i] > r3_aligned[i] or not weekly_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long setup: price at S3/S4 with bullish weekly trend
                if weekly_uptrend and (abs(close[i] - s3_aligned[i]) < 0.001 * close[i] or 
                                       abs(close[i] - s4_aligned[i]) < 0.001 * close[i]):
                    position = 1
                    signals[i] = 0.25
                # Short setup: price at R3/R4 with bearish weekly trend
                elif weekly_downtrend and (abs(close[i] - r3_aligned[i]) < 0.001 * close[i] or 
                                           abs(close[i] - r4_aligned[i]) < 0.001 * close[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals