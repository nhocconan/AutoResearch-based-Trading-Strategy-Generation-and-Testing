#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h_1d_camarilla_breakout_v4
# Uses daily Camarilla pivot levels (H3/L3) on the 12-hour chart with volume confirmation.
# Long when price closes above H3 with volume > 1.5x 20-period average.
# Short when price closes below L3 with volume > 1.5x 20-period average.
# Exits when price returns to daily pivot point (PP).
# Uses 12h timeframe to target 12-37 trades/year (50-150 total) to minimize fee drag.
# Works in bull markets via breakouts and bear markets via mean reversion to pivot.

name = "12h_1d_camarilla_breakout_v4"
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
    
    # Calculate daily Camarilla levels based on previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and Camarilla levels for each day
    pp = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: H3 = PP + 1.1/4 * range, L3 = PP - 1.1/4 * range
    h3 = pp + (1.1 / 4) * range_1d
    l3 = pp - (1.1 / 4) * range_1d
    
    # Align daily levels to 12h timeframe (daily values update after daily bar closes)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Volume confirmation: volume > 1.5 * 20-period average (12h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
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
        
        # Long signal: price closes above H3
        if close[i] > h3_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price closes below L3
        elif close[i] < l3_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: price returns to daily pivot point (mean reversion)
        elif position == 1 and close[i] <= pp_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= pp_aligned[i]:
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