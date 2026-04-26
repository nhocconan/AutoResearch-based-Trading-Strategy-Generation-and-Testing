#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_12hTrend_1dVolumeSpike_v1
Hypothesis: 6h Donchian(20) breakout with 12h EMA50 trend filter and 1d volume spike confirmation. 
Donchian breakouts capture momentum in both bull and bear markets. The 12h EMA50 ensures we only 
trade in the direction of the intermediate trend, reducing whipsaws. 1d volume spike (2.0x 20-period 
average) confirms institutional participation. Discrete position sizing (0.25) limits drawdown. 
Target: 20-50 trades/year per symbol to minimize fee drag in ranging markets like 2025+.
"""

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
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d volume spike: volume > 2.0 * 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (2.0 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate Donchian channels on 6h (primary timeframe)
    # Donchian(20): upper = max(high, 20), lower = min(low, 20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 12h EMA(50), 1d volume MA, Donchian(20)
    start_idx = max(50, 20, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(volume_spike_1d_aligned[i]) or
            np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        trend_12h_up = close_val > ema_50_12h_aligned[i]   # 12h uptrend
        trend_12h_down = close_val < ema_50_12h_aligned[i]  # 12h downtrend
        vol_spike = volume_spike_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper AND 12h trend up AND volume spike
            long_signal = (close_val > donchian_upper[i]) and trend_12h_up and vol_spike
            
            # Short: price breaks below Donchian lower AND 12h trend down AND volume spike
            short_signal = (close_val < donchian_lower[i]) and trend_12h_down and vol_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: trend flips down OR price re-enters Donchian channel (mean reversion)
            if (not trend_12h_up) or (close_val < donchian_upper[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend flips up OR price re-enters Donchian channel
            if (not trend_12h_down) or (close_val > donchian_lower[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_12hTrend_1dVolumeSpike_v1"
timeframe = "6h"
leverage = 1.0