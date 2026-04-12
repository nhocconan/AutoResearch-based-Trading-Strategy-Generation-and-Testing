#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h_1d_camarilla_pivot_breakout_volume
# Uses daily Camarilla pivot levels (based on prior day's high/low/close) as entry/exit levels on 12h chart.
# Long when price breaks above R4 with volume confirmation (volume > 1.5x 20-period avg).
# Short when price breaks below S4 with volume confirmation.
# Exits when price returns to the prior day's close (pivot point).
# Designed for low trade frequency (target: 12-37 trades/year) to minimize fee drift.
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
    
    # Calculate daily Camarilla levels (based on prior day's high/low/close)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior day's values (shifted by 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # First day has no prior data
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Pivot point (prior day's close)
    pivot = prev_close
    
    # Camarilla levels: R4, R3, R2, R1, S1, S2, S3, S4
    # R4 = Close + 1.5*(High - Low)
    # R3 = Close + 1.1*(High - Low)
    # R2 = Close + 0.6*(High - Low)
    # R1 = Close + 0.3*(High - Low)
    # S1 = Close - 0.3*(High - Low)
    # S2 = Close - 0.6*(High - Low)
    # S3 = Close - 0.9*(High - Low)
    # S4 = Close - 1.2*(High - Low)
    # Using prior day's range
    high_low_diff = prev_high - prev_low
    r4 = prev_close + 1.5 * high_low_diff
    s4 = prev_close - 1.2 * high_low_diff
    # Exit at pivot (prior day's close)
    
    # Align daily Camarilla levels to 12h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Volume confirmation: volume > 1.5 * 20-period average (12h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(pivot_aligned[i]):
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
        
        # Long signal: price breaks above R4
        if close[i] > r4_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below S4
        elif close[i] < s4_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: price returns to pivot (mean reversion)
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