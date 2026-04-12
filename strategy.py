#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d_1w_camarilla_breakout_v2
# Uses weekly Camarilla pivot levels (H3/L3) as key support/resistance on daily chart.
# Long when price breaks above H3 with volume confirmation (volume > 1.5x 20-day avg).
# Short when price breaks below L3 with volume confirmation.
# Exits when price returns to weekly pivot point (PP).
# Uses tighter H3/L3 levels (vs H4/L4) for more frequent but still controlled breakouts.
# Designed for low trade frequency (target: 20-30 trades/year) to minimize fee drift.
# Works in trending markets via breakouts and in ranging markets via mean reversion to pivot.

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
    
    # Calculate weekly Camarilla levels
    # Based on previous week's OHLC
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot point and Camarilla levels for each week
    pp = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    # Camarilla levels: H3 = PP + 1.1/4 * range, L3 = PP - 1.1/4 * range
    h3 = pp + (1.1 / 4) * range_1w
    l3 = pp - (1.1 / 4) * range_1w
    
    # Align weekly levels to daily timeframe (weekly values update after weekly bar closes)
    h3_aligned = align_htf_to_ltf(prices, df_1w, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1w, l3)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    
    # Volume confirmation: volume > 1.5 * 20-period average (daily timeframe)
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
        
        # Long signal: price breaks above H3
        if close[i] > h3_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below L3
        elif close[i] < l3_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: price returns to weekly pivot point (mean reversion)
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