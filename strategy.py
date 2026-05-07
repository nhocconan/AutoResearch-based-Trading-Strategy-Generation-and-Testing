#!/usr/bin/env python3
"""
4h_Donchian_Breakout_1dTrend_Volume
Hypothesis: On 4h timeframe, price breaking above Donchian(20) high or below Donchian(20) low,
combined with 1d EMA34 trend filter and volume confirmation (current volume > 1.5x 20-period average),
captures momentum moves with controlled frequency. The 1d trend filter ensures alignment with higher
timeframe momentum, reducing false breakouts in choppy conditions. Designed for 20-40 trades/year per symbol.
"""
name = "4h_Donchian_Breakout_1dTrend_Volume"
timeframe = "4h"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 4h Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # wait for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_avg[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + 1d uptrend + volume
            if close[i] > high_max[i] and close[i] > ema_34_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + 1d downtrend + volume
            elif close[i] < low_min[i] and close[i] < ema_34_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price crosses back through the opposite Donchian level
            if position == 1:
                if close[i] < low_min[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > high_max[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals