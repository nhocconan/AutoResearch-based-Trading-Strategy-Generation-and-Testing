#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1d_donchian_breakout_volume
# Uses daily Donchian channels (20-period high/low) as breakout levels on 4h chart.
# Long when price breaks above 20-day high with volume confirmation (volume > 1.5x 20-period avg).
# Short when price breaks below 20-day low with volume confirmation.
# Exits when price crosses the 20-day midpoint (mean reversion).
# Designed for low trade frequency (target: 20-50 trades/year) to minimize fee drag.
# Works in trending markets via breakouts and ranging markets via mean reversion to midpoint.
# Focus on BTC/ETH as primary targets.

name = "4h_1d_donchian_breakout_volume"
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
    
    # Get daily data for Donchian channel calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 20-period high and low
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    # Midpoint for exit
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Align daily Donchian levels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    
    # Volume confirmation: volume > 1.5 * 20-period average (4h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(donchian_mid_aligned[i]):
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
        if close[i] > donchian_high_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below 20-day low
        elif close[i] < donchian_low_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: price crosses 20-day midpoint (mean reversion)
        elif position == 1 and close[i] <= donchian_mid_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= donchian_mid_aligned[i]:
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