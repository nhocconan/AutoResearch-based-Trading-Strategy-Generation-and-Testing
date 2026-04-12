#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_12h_camarilla_pivot_volume
# Uses daily Camarilla pivot levels (from 12h data) as support/resistance on 4h chart.
# Long when price touches S3 level with volume confirmation (volume > 1.5x 20-period avg).
# Short when price touches R3 level with volume confirmation.
# Exits when price crosses the central pivot (mean reversion to equilibrium).
# Designed for low trade frequency (target: 20-50 trades/year) to minimize fee drag.
# Works in trending markets via breakouts and ranging markets via mean reversion to pivot.
# Focus on BTC/ETH as primary targets.

name = "4h_12h_camarilla_pivot_volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels from 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Daily pivot calculation (using previous day's OHLC)
    pivot = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    
    # Camarilla levels
    r3 = pivot + (range_12h * 1.1 / 4)
    s3 = pivot - (range_12h * 1.1 / 4)
    r4 = pivot + (range_12h * 1.1 / 2)
    s4 = pivot - (range_12h * 1.1 / 2)
    
    # Align 12h Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot)
    
    # Volume confirmation: volume > 1.5 * 20-period average (4h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(pivot_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Require volume confirmation for new entries
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price touches S3 level (strong support)
        if low[i] <= s3_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price touches R3 level (strong resistance)
        elif high[i] >= r3_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: price crosses central pivot (mean reversion)
        elif position == 1 and close[i] >= pivot_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] <= pivot_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals