#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hTrend_VolumeS
Hypothesis: On 4h timeframe, enter long when price breaks above 20-period Donchian high with volume confirmation and 12h uptrend (EMA50 > EMA100), enter short when price breaks below 20-period Donchian low with volume confirmation and 12h downtrend. Exit on opposite breakout with volume. Designed for 20-50 trades/year to minimize fee drag while capturing trends in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 100:
        return np.zeros(n)
    
    # Calculate 12h 50 and 100 EMA for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema100_12h = pd.Series(close_12h).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # Align 12h EMAs to 4h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    ema100_12h_aligned = align_htf_to_ltf(prices, df_12h, ema100_12h)
    
    # 12h trend: bullish when EMA50 > EMA100
    trend_12h_up = ema50_12h_aligned > ema100_12h_aligned
    trend_12h_down = ema50_12h_aligned < ema100_12h_aligned
    
    # Calculate 20-period Donchian channels (using previous 20 periods, not including current)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: current volume > 1.5x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_surge = volume > (vol_ma_50 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(ema100_12h_aligned[i]) or
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with 12h trend alignment and volume surge
        long_entry = close[i] > high_20[i] and trend_12h_up[i] and volume_surge[i]
        short_entry = close[i] < low_20[i] and trend_12h_down[i] and volume_surge[i]
        
        # Exit on opposite 20-period Donchian break with volume surge
        long_exit = close[i] < low_20[i] and volume_surge[i]
        short_exit = close[i] > high_20[i] and volume_surge[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_Breakout_12hTrend_VolumeS"
timeframe = "4h"
leverage = 1.0