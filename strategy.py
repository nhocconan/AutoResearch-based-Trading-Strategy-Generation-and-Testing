#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d_1w_camarilla_pivot_breakout
# Uses weekly Camarilla pivot levels on 1d chart for breakouts in trending markets
# and mean-reversion at pivot levels in ranging markets.
# Long when price breaks above weekly H4 with volume > 1.5x 20-day average.
# Short when price breaks below weekly L4 with volume confirmation.
# Exit when price touches weekly H3/L3 (mean reversion) or crosses weekly pivot.
# Designed for low trade frequency (target: 15-30 trades/year) to minimize fee drag.
# Works in trending markets via breakouts and ranging markets via mean reversion.
# Focus on BTC/ETH as primary targets.

name = "1d_1w_camarilla_pivot_breakout"
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
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    # Weekly range
    weekly_range = high_1w - low_1w
    
    # Camarilla levels (based on weekly data)
    # H4 = pivot + 1.5 * range
    # L4 = pivot - 1.5 * range
    # H3 = pivot + 1.25 * range
    # L3 = pivot - 1.25 * range
    # H2 = pivot + 1.083 * range
    # L2 = pivot - 1.083 * range
    # H1 = pivot + 0.833 * range
    # L1 = pivot - 0.833 * range
    
    camarilla_h4 = weekly_pivot + 1.5 * weekly_range
    camarilla_l4 = weekly_pivot - 1.5 * weekly_range
    camarilla_h3 = weekly_pivot + 1.25 * weekly_range
    camarilla_l3 = weekly_pivot - 1.25 * weekly_range
    camarilla_pivot = weekly_pivot
    
    # Align weekly Camarilla levels to 1d timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pivot)
    
    # Volume confirmation: volume > 1.5 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or np.isnan(camarilla_pivot_aligned[i]):
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
        
        # Long signal: price breaks above weekly H4
        if close[i] > camarilla_h4_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below weekly L4
        elif close[i] < camarilla_l4_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: mean reversion at H3/L3 or pivot
        elif position == 1 and (close[i] <= camarilla_h3_aligned[i] or close[i] <= camarilla_pivot_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] >= camarilla_l3_aligned[i] or close[i] >= camarilla_pivot_aligned[i]):
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