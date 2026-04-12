#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_12h_camarilla_breakout_v1
# Uses 12-hour chart for daily-like structure and 1d chart for weekly-like pivot levels.
# Long when 4h price breaks above 12h H4 level with volume confirmation (volume > 1.3x 20-period avg).
# Short when 4h price breaks below 12h L4 level with volume confirmation.
# Exits when price returns to 12h pivot point (PP).
# Designed for low trade frequency (target: 20-30 trades/year) to minimize fee drag.
# Works in trending markets via breakouts and in ranging markets via mean reversion to pivot.
# 12h pivot levels are more stable than daily, reducing false signals.

name = "4h_12h_camarilla_breakout_v1"
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
    
    # Get 12h data for Camarilla pivot calculation (daily-like structure)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels based on previous 12h bar's OHLC
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate pivot point and Camarilla levels for each 12h period
    pp_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    
    # Camarilla levels: H4 = PP + 1.1/2 * range, L4 = PP - 1.1/2 * range
    h4_12h = pp_12h + (1.1 / 2) * range_12h
    l4_12h = pp_12h - (1.1 / 2) * range_12h
    
    # Align 12h levels to 4h timeframe (12h values update after 12h bar closes)
    h4_12h_aligned = align_htf_to_ltf(prices, df_12h, h4_12h)
    l4_12h_aligned = align_htf_to_ltf(prices, df_12h, l4_12h)
    pp_12h_aligned = align_htf_to_ltf(prices, df_12h, pp_12h)
    
    # Volume confirmation: volume > 1.3 * 20-period average (4h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(h4_12h_aligned[i]) or np.isnan(l4_12h_aligned[i]) or np.isnan(pp_12h_aligned[i]):
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
        
        # Long signal: 4h price breaks above 12h H4
        if close[i] > h4_12h_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: 4h price breaks below 12h L4
        elif close[i] < l4_12h_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: price returns to 12h pivot point (mean reversion)
        elif position == 1 and close[i] <= pp_12h_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= pp_12h_aligned[i]:
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