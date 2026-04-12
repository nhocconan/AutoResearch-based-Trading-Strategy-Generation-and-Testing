#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h_1d_1w_camarilla_breakout_v3
# Uses weekly and daily pivot levels as dynamic support/resistance on 12h chart.
# Long when price breaks above daily H4 with volume confirmation (>1.5x 20-period avg) and weekly trend alignment (price > weekly P).
# Short when price breaks below daily L4 with volume confirmation and weekly trend alignment (price < weekly P).
# Exits when price returns to daily pivot point (mean reversion).
# Designed for low trade frequency (target: 15-30 trades/year) to minimize fee drag.
# Works in trending markets via breakouts and in ranging markets via mean reversion to pivot.
# Uses 12h timeframe to balance signal frequency and noise reduction.

name = "12h_1d_1w_camarilla_breakout_v3"
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
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and Camarilla levels for each day
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: H4 = PP + 1.1/2 * range, L4 = PP - 1.1/2 * range
    h4_1d = pp_1d + (1.1 / 2) * range_1d
    l4_1d = pp_1d - (1.1 / 2) * range_1d
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot point (based on previous week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    
    # Align daily levels to 12h timeframe
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    
    # Volume confirmation: volume > 1.5 * 20-period average (12h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or np.isnan(pp_1d_aligned[i]) or np.isnan(pp_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Require volume confirmation and weekly trend alignment for new entries
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above daily H4 AND price above weekly pivot (uptrend)
        if close[i] > h4_1d_aligned[i] and close[i] > pp_1w_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below daily L4 AND price below weekly pivot (downtrend)
        elif close[i] < l4_1d_aligned[i] and close[i] < pp_1w_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: price returns to daily pivot point (mean reversion)
        elif position == 1 and close[i] <= pp_1d_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= pp_1d_aligned[i]:
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