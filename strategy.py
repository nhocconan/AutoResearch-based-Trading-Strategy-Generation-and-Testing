#!/usr/bin/env python3
"""
6h_LarryWilliamsVolBreakout_12hTrend_1dVolumeFilter
Hypothesis: Larry Williams Volatility Breakout combined with 12h EMA trend filter and 1d volume spike. 
Long: open > previous high + K*(previous high - previous low) where K=0.5, with 12h EMA up and volume spike.
Short: open < previous low - K*(previous high - previous low) where K=0.5, with 12h EMA down and volume spike.
Uses 6h chart for entry timing, 12h for trend, 1d for volume confirmation.
Designed to work in both bull (breakouts) and bear (breakdowns) markets with controlled trade frequency.
Target: 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # 12h EMA21 for trend filter
    close_12h = pd.Series(df_12h['close'].values)
    ema21_12h = close_12h.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema21_12h)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d volume 20-period average for spike detection
    volume_1d = pd.Series(df_1d['volume'].values)
    vol_ma_20_1d = volume_1d.rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Williams Volatility Breakout calculation on 6h data
    # K = 0.5 (standard Williams %R parameter)
    K = 0.5
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_range = prev_high - prev_low
    
    # Long breakout: open > previous high + K * previous range
    long_breakout_level = prev_high + K * prev_range
    # Short breakdown: open < previous low - K * previous range
    short_breakout_level = prev_low - K * prev_range
    
    # Volume filter: current 6h volume > 1.5x 1d average volume (scaled)
    # Convert 1d average volume to 6h equivalent (1d = 4 * 6h)
    vol_threshold = vol_ma_20_1d_aligned * 1.5 / 4.0  # Scale down for 6h comparison
    volume_filter = volume > vol_threshold
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period (need 1 for roll, 20 for volume MA, 21 for EMA)
    start_idx = max(1, 20, 21)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema21_12h_aligned[i]) or np.isnan(long_breakout_level[i]) or 
            np.isnan(short_breakout_level[i]) or np.isnan(volume_filter[i]) or
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: open breaks above previous high + K*range with trend up and volume spike
            if (open_price[i] > long_breakout_level[i] and 
                ema21_12h_aligned[i] > ema21_12h_aligned[i-1] and  # 12h EMA rising
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: open breaks below previous low - K*range with trend down and volume spike
            elif (open_price[i] < short_breakout_level[i] and 
                  ema21_12h_aligned[i] < ema21_12h_aligned[i-1] and  # 12h EMA falling
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price closes below 12h EMA or reversal signal
            if close[i] < ema21_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above 12h EMA or reversal signal
            if close[i] > ema21_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_LarryWilliamsVolBreakout_12hTrend_1dVolumeFilter"
timeframe = "6h"
leverage = 1.0