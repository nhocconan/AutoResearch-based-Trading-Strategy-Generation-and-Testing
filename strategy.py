#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d_1w_camarilla_breakout_v2
# Uses weekly Camarilla pivot levels (H3/L3) as key support/resistance on daily chart.
# Long when price breaks above H3 with volume confirmation (volume > 1.3x 20-day avg).
# Short when price breaks below L3 with volume confirmation.
# Exits when price crosses the weekly pivot point (PP) in opposite direction.
# H3/L3 levels are more sensitive than H4/L4, increasing signal frequency while maintaining reliability.
# Weekly pivot provides mean-reversion target in ranging markets.
# Designed for 15-25 trades/year to minimize fee drag while capturing trends and reversals.

name = "1d_1w_camarilla_breakout_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels (using previous week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point and range
    pp = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    # Camarilla H3/L3 levels (more sensitive than H4/L4)
    # H3 = PP + 1.1/4 * range, L3 = PP - 1.1/4 * range
    h3 = pp + (1.1 / 4) * range_1w
    l3 = pp - (1.1 / 4) * range_1w
    
    # Align weekly levels to daily timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1w, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1w, l3)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    
    # Volume confirmation: volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(pp_aligned[i]):
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
        
        # Long signal: price breaks above H3
        if close[i] > h3_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below L3
        elif close[i] < l3_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: price crosses weekly pivot point in opposite direction
        elif position == 1 and close[i] < pp_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > pp_aligned[i]:
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