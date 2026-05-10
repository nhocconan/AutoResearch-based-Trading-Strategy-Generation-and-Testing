#!/usr/bin/env python3
# 12h_ThreeD_Squeeze_Breakout_With_VolumeSpike
# Hypothesis: In BTC/ETH markets, volatility contractions followed by breakouts with volume confirmation
# capture strong moves in both bull and bear markets. Uses 1-day Bollinger Band width for volatility
# regime detection and 12-hour price action for breakout signals. Low trade frequency (~20-40/year)
# minimizes fee impact while capturing explosive moves.

name = "12h_ThreeD_Squeeze_Breakout_With_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtd_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Bollinger Bands on daily chart
    close_1d = pd.Series(df_1d['close'])
    bb_middle = close_1d.rolling(window=20, min_periods=20).mean().values
    bb_std = close_1d.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle  # Normalized width
    
    # Align BB width to 12h timeframe
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    
    # Calculate 12h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (24-period MA = 12 days)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need BB width (20), Donchian (20), volume MA (24)
    start_idx = max(20, 20, 24)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(bb_width_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility squeeze condition: BB width at 20-period low
        # Look back 20 periods to find minimum
        if i >= 40:  # Need 20 + 20 for lookback
            bb_width_min = np.min(bb_width_aligned[i-20:i])
            squeeze = bb_width_aligned[i] <= bb_width_min * 1.1  # Within 10% of minimum
        else:
            squeeze = False
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Breakout conditions
        breakout_up = close[i] > donchian_high[i]
        breakout_down = close[i] < donchian_low[i]
        
        if position == 0:
            # Long entry: volatility squeeze + upward breakout + volume
            if squeeze and breakout_up and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: volatility squeeze + downward breakout + volume
            elif squeeze and breakout_down and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: breakdown below Donchian low or volatility expansion
            if close[i] < donchian_low[i] or bb_width_aligned[i] > bb_width_min * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: breakout above Donchian high or volatility expansion
            if close[i] > donchian_high[i] or bb_width_aligned[i] > bb_width_min * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals