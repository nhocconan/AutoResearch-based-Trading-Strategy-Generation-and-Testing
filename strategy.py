#!/usr/bin/env python3
"""
12h_1d_Weekly_Camarilla_Pivot_Breakout_Volume_Confirmation
Hypothesis: Weekly (1w) timeframe provides stronger trend context than daily.
Breakouts above weekly R3 or below weekly S3 on 12h chart with volume expansion capture
major institutional moves while avoiding minor false breakouts. Uses 1d for volume context
to avoid overnight gaps. Designed for 12-37 trades/year to stay within fee limits.
Works in bull markets (breakouts continuation) and bear markets (breakdowns continuation).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 300:  # Need sufficient history for weekly and daily calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for primary trend context (more reliable than daily)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for volume confirmation (avoid overnight gaps)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Weekly Camarilla levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    close_prev_1w = np.roll(close_1w, 1)
    close_prev_1w[0] = close_1w[0]  # first bar uses its own close
    
    range_1w = high_1w - low_1w
    
    # Weekly Resistance (R3) and Support (S3)
    R3_1w = close_prev_1w + (range_1w * 1.2500 / 4)
    S3_1w = close_prev_1w - (range_1w * 1.2500 / 4)
    
    # Align weekly levels to 12h timeframe
    R3_1w_aligned = align_htf_to_ltf(prices, df_1w, R3_1w)
    S3_1w_aligned = align_htf_to_ltf(prices, df_1w, S3_1w)
    
    # Calculate Daily Average Volume for confirmation (avoid overnight gaps)
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean()
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d.values)
    
    # Volume expansion: current volume > 1.8x 20-day average volume
    volume_expansion = volume > (vol_ma_20_1d_aligned * 1.8)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size to manage drawdown
    
    for i in range(300, n):  # Start after sufficient warmup
        # Skip if any required data is not ready
        if (np.isnan(R3_1w_aligned[i]) or np.isnan(S3_1w_aligned[i]) or 
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long breakout: price breaks above weekly R3 with volume expansion
        long_breakout = close[i] > R3_1w_aligned[i] and volume_expansion[i]
        
        # Short breakdown: price breaks below weekly S3 with volume expansion
        short_breakout = close[i] < S3_1w_aligned[i] and volume_expansion[i]
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "12h_1d_Weekly_Camarilla_Pivot_Breakout_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0