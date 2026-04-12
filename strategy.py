#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h_1d_camarilla_pivot_breakout_volume
# Uses daily Camarilla pivot levels (based on previous day's range) as support/resistance.
# Long when price breaks above H4 level with volume confirmation (volume > 1.5x 20-period avg).
# Short when price breaks below L4 level with volume confirmation.
# Exits when price crosses the daily pivot (mean reversion).
# Designed for low trade frequency (target: 12-37 trades/year) on 12h timeframe.
# Works in trending markets via breakouts and ranging markets via mean reversion to pivot.
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
    
    # Calculate daily Camarilla pivot levels (based on previous day's range)
    # H4 = Close + 1.5 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    # Pivot = (High + Low + Close) / 3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_high = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_low = close_1d - 1.5 * (high_1d - low_1d)
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3.0
    
    # Align daily Camarilla levels to 12h timeframe
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Volume confirmation: volume > 1.5 * 20-period average (12h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or np.isnan(camarilla_pivot_aligned[i]):
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
        
        # Long signal: price breaks above H4 level
        if close[i] > camarilla_high_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below L4 level
        elif close[i] < camarilla_low_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: price crosses daily pivot (mean reversion)
        elif position == 1 and close[i] <= camarilla_pivot_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= camarilla_pivot_aligned[i]:
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