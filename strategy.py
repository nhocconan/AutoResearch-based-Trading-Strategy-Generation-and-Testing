#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h_1d_donchian_breakout_v1
# Uses daily Donchian channels (20-day high/low) for breakout signals on 12h chart.
# Long when price breaks above 20-day high with volume confirmation (volume > 1.5x 20-period avg).
# Short when price breaks below 20-day low with volume confirmation.
# Exits when price returns to 10-day moving average (trend fade).
# Designed for low trade frequency (target: 15-25 trades/year) to minimize fee drag.
# Works in trending markets via breakouts and in ranging markets via mean reversion to mean.

name = "12h_1d_donchian_breakout_v1"
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
    
    # Get daily data for Donchian channel calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-period high/low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period high and low for Donchian channels
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align daily Donchian levels to 12h timeframe (daily values update after daily bar closes)
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Calculate 10-day moving average for exit (trend fade)
    ma_10 = pd.Series(close_1d).rolling(window=10, min_periods=10).mean().values
    ma_10_aligned = align_htf_to_ltf(prices, df_1d, ma_10)
    
    # Volume confirmation: volume > 1.5 * 20-period average (12h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or np.isnan(ma_10_aligned[i]):
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
        
        # Long signal: price breaks above 20-day high
        if close[i] > high_20_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below 20-day low
        elif close[i] < low_20_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: price returns to 10-day moving average (trend fade)
        elif position == 1 and close[i] <= ma_10_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= ma_10_aligned[i]:
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