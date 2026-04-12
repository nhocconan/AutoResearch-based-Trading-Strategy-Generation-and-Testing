#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h_1d_camarilla_breakout_volume
# Uses daily Camarilla pivot levels (based on previous day's range) as support/resistance on 12h chart.
# Long when price breaks above H4 level with volume confirmation (volume > 1.5x 20-period avg).
# Short when price breaks below L4 level with volume confirmation.
# Exits when price crosses the pivot point (mean reversion).
# Camarilla levels are designed for intraday trading but work well on higher timeframes.
# Focus on BTC/ETH as primary targets.

name = "12h_1d_camarilla_breakout_volume"
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
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    # Based on previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shift by 1 to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # First value will be invalid (rolled from last), but we'll handle with min_periods equivalent
    
    # Camarilla calculations
    # Pivot point
    pivot = (prev_high + prev_low + prev_close) / 3.0
    # Range
    range_val = prev_high - prev_low
    
    # Resistance levels
    r4 = prev_close + range_val * 1.500
    r3 = prev_close + range_val * 1.250
    r2 = prev_close + range_val * 1.166
    r1 = prev_close + range_val * 1.083
    
    # Support levels
    s1 = prev_close - range_val * 1.083
    s2 = prev_close - range_val * 1.166
    s3 = prev_close - range_val * 1.250
    s4 = prev_close - range_val * 1.500
    
    # Primary levels for breakout: H4 (r4) and L4 (s4)
    camarilla_high = r4
    camarilla_low = s4
    camarilla_pivot = pivot
    
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
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i])):
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
        # Exit conditions: price crosses pivot point (mean reversion)
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