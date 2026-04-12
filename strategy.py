#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h_1d_camarilla_pivot_breakout_volume
# Uses daily Camarilla pivot levels (support/resistance) as breakout/mean-reversion levels on 12h chart.
# Long when price breaks above Camarilla H4 resistance with volume confirmation (>1.5x 20-period avg volume).
# Short when price breaks below Camarilla L4 support with volume confirmation.
# Exits when price returns to Camarilla Pivot level (mean reversion).
# Works in trending markets via breakouts and ranging markets via mean reversion to pivot.
# Volume confirmation reduces false breakouts. Designed for low trade frequency (target: 15-35/year) to minimize fee drag.
# Focus on BTC/ETH as primary targets.

name = "12h_1d_camarilla_pivot_breakout_volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_ = high_1d - low_1d
    
    # Resistance levels
    r1 = close_1d + (range_ * 1.1 / 12)
    r2 = close_1d + (range_ * 1.1 / 6)
    r3 = close_1d + (range_ * 1.1 / 4)
    r4 = close_1d + (range_ * 1.1 / 2)  # Strongest resistance
    
    # Support levels
    s1 = close_1d - (range_ * 1.1 / 12)
    s2 = close_1d - (range_ * 1.1 / 6)
    s3 = close_1d - (range_ * 1.1 / 4)
    s4 = close_1d - (range_ * 1.1 / 2)  # Strongest support
    
    # Align daily Camarilla levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: volume > 1.5 * 20-period average (12h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(pivot_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]):
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
        
        # Long signal: price breaks above Camarilla H4 resistance
        if close[i] > r4_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below Camarilla L4 support
        elif close[i] < s4_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: price returns to Camarilla Pivot (mean reversion)
        elif position == 1 and close[i] <= pivot_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= pivot_aligned[i]:
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