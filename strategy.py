#!/usr/bin/env python3
"""
1d_1W_Camarilla_R1S1_Breakout_Volume_Momentum_v1
Hypothesis: Use weekly price structure for directional bias, enter on daily close above weekly resistance or below weekly support with volume confirmation. Weekly timeframe reduces noise, daily entries capture swings. Volume filter ensures conviction. Designed to work in both bull and bear markets by focusing on significant breaks.
Target: 10-20 trades/year per symbol (40-80 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for directional bias
    df_1w = get_htf_data(prices, '1w')
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Previous week's OHLC for weekly support/resistance
    prev_close = np.roll(close_1w, 1)
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close[0] = close_1w[0]
    prev_high[0] = high_1w[0]
    prev_low[0] = low_1w[0]
    
    # Weekly range and key levels
    weekly_range = prev_high - prev_low
    resistance = prev_close + weekly_range * 1.1 / 12  # Weekly R1
    support = prev_close - weekly_range * 1.1 / 12     # Weekly S1
    
    # Align weekly levels to daily timeframe
    resistance_aligned = align_htf_to_ltf(prices, df_1w, resistance)
    support_aligned = align_htf_to_ltf(prices, df_1w, support)
    
    # Daily volume confirmation: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(resistance_aligned[i]) or np.isnan(support_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: daily close above weekly resistance with volume
            if close[i] > resistance_aligned[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: daily close below weekly support with volume
            elif close[i] < support_aligned[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: daily close back below weekly resistance or volume fails
            if close[i] < resistance_aligned[i] or not vol_confirm[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: daily close back above weekly support or volume fails
            if close[i] > support_aligned[i] or not vol_confirm[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1W_Camarilla_R1S1_Breakout_Volume_Momentum_v1"
timeframe = "1d"
leverage = 1.0