#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike
Hypothesis: Trade 4h Donchian(20) breakouts with 12h EMA50 trend filter and volume confirmation.
Donchian breakouts capture strong momentum moves, while 12h EMA50 ensures we trade with the higher timeframe trend.
Volume confirmation filters out weak breakouts. Works in both bull and bear markets by following the 12h trend.
Targets 75-200 total trades over 4 years (19-50/year) to minimize fee drag.
Uses discrete position sizing (0.0, ±0.30) to balance performance and fee churn.
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
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 20-period volume SMA on 4h for volume confirmation
    # Using rolling window with min_periods
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period high/low) on 4h
    # Using rolling window with min_periods
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of Donchian(20), volume SMA(20), and 12h EMA50 alignment
    start_idx = max(20, 20)  # Donchian and volume SMA both need 20 periods
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(high_max_20[i]) or
            np.isnan(low_min_20[i]) or
            np.isnan(volume_sma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        close_val = close[i]
        vol_ma = volume_sma_20[i]
        
        # Volume confirmation: current volume > 1.5 x 20-period average
        volume_confirm = volume[i] > (1.5 * vol_ma)
        
        if position == 0:
            # Long: price breaks above 20-period high AND 12h trend up AND volume confirmation
            long_signal = (close_val > high_max_20[i]) and (close_val > ema_50_12h_aligned[i]) and volume_confirm
            
            # Short: price breaks below 20-period low AND 12h trend down AND volume confirmation
            short_signal = (close_val < low_min_20[i]) and (close_val < ema_50_12h_aligned[i]) and volume_confirm
            
            if long_signal:
                signals[i] = 0.30
                position = 1
            elif short_signal:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            # Exit: price breaks below 20-period low OR 12h trend flips down
            if (close_val < low_min_20[i]) or (close_val < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            # Exit: price breaks above 20-period high OR 12h trend flips up
            if (close_val > high_max_20[i]) or (close_val > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0