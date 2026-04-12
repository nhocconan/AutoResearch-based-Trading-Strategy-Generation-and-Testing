#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1d_donchian_breakout_volume_v1
# Uses 20-day Donchian channel breakout with volume confirmation and ATR filter.
# Long when price breaks above upper band with volume > 1.5x 20-day avg.
# Short when price breaks below lower band with volume confirmation.
# Exits when price returns to 20-day moving average.
# Designed for 4h timeframe to capture medium-term trends with low trade frequency.
# Works in trending markets via breakouts and includes volume filter to reduce false signals.

name = "4h_1d_donchian_breakout_volume_v1"
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
    
    # Get daily data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-day Donchian channel (using daily high/low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 20-period high and low for Donchian channels
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align daily Donchian levels to 4h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # Calculate 20-day moving average for exit (using daily close)
    close_1d = df_1d['close'].values
    ma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    ma_20_aligned = align_htf_to_ltf(prices, df_1d, ma_20)
    
    # Volume confirmation: volume > 1.5 * 20-period average (4h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or np.isnan(ma_20_aligned[i]):
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
        
        # Long signal: price breaks above upper Donchian band
        if close[i] > upper_20_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below lower Donchian band
        elif close[i] < lower_20_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: price returns to 20-day moving average
        elif position == 1 and close[i] <= ma_20_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= ma_20_aligned[i]:
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