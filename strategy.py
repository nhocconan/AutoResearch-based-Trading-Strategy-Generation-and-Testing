#!/usr/bin/env python3
"""
6h_pivot_breakout_1w_trend_volume_v1
Hypothesis: Breakout above/below weekly pivot R4/S4 levels with volume confirmation and daily trend filter.
Trades with the daily trend (using EMA50) for momentum continuation.
Weekly pivot levels provide strong support/resistance; breaks indicate institutional interest.
Volume confirms conviction. Targets 15-35 trades/year by requiring:
- Price breaks weekly R4 (for long) or S4 (for short)
- Volume > 2.0x 20-period average
- Price > EMA50 (long) or < EMA50 (short) for trend alignment
Works in bull markets (breaks continue up) and bear markets (breaks continue down).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_pivot_breakout_1w_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot points (R4/S4)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    pivot_w = (high_w + low_w + close_w) / 3.0
    # R4 = P + 3*(H - L), S4 = P - 3*(H - L)
    r4_w = pivot_w + 3.0 * (high_w - low_w)
    s4_w = pivot_w - 3.0 * (high_w - low_w)
    
    # Daily data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Align weekly R4/S4 to 6h timeframe
    r4_w_6h = align_htf_to_ltf(prices, df_1w, r4_w)
    s4_w_6h = align_htf_to_ltf(prices, df_1w, s4_w)
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema50_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 20-period volume average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema50_6h[i]) or 
            np.isnan(r4_w_6h[i]) or 
            np.isnan(s4_w_6h[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x average volume
        vol_confirm = volume[i] > 2.0 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below weekly S4 OR trend turns down
            if close[i] < s4_w_6h[i] or close[i] < ema50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above weekly R4 OR trend turns up
            if close[i] > r4_w_6h[i] or close[i] > ema50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above weekly R4 + volume + uptrend
            if (close[i] > r4_w_6h[i] and 
                vol_confirm and 
                close[i] > ema50_6h[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below weekly S4 + volume + downtrend
            elif (close[i] < s4_w_6h[i] and 
                  vol_confirm and 
                  close[i] < ema50_6h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals