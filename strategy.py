#!/usr/bin/env python3
name = "1d_Weekly_Channel_Breakout_With_Volume_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Donchian channels (20-week period)
    high_1w = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    low_1w = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    
    # Align to daily - weekly signals available after weekly bar close
    high_1w_aligned = align_htf_to_ltf(prices, df_1w, high_1w)
    low_1w_aligned = align_htf_to_ltf(prices, df_1w, low_1w)
    
    # Daily volume filter: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian channels
    
    for i in range(start_idx, n):
        if np.isnan(high_1w_aligned[i]) or np.isnan(low_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above weekly high with volume confirmation
            if close[i] > high_1w_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below weekly low with volume confirmation
            elif close[i] < low_1w_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price breaks below weekly low
            if close[i] < low_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price breaks above weekly high
            if close[i] > high_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly Donchian breakout with volume confirmation on daily timeframe.
# Long when price breaks above 20-week high with above-average volume.
# Short when price breaks below 20-week low with above-average volume.
# Uses weekly timeframe for structure to avoid whipsaws, daily for execution.
# Volume filter ensures breakout conviction. Position size 0.25 limits risk.
# Works in trending markets (breakouts) and avoids choppy periods.
# Target: 10-25 trades/year to minimize fee decay while capturing major moves.