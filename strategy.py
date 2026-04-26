#!/usr/bin/env python3
"""
6h_WeeklyDonchianBreakout_1dTrendFilter_VolumeSpike
Hypothesis: 6h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation.
Enters long when price breaks above 20-period high AND close > 1d EMA50 AND volume > 2.0 * 20-period average volume.
Enters short when price breaks below 20-period low AND close < 1d EMA50 AND volume > 2.0 * 20-period average volume.
Exits on opposite Donchian breakout or when price re-enters the 20-period range.
Uses 1d EMA50 for higher timeframe trend alignment to avoid counter-trend trades.
Volume spike (2.0x) confirms strong participation. Designed for 6h timeframe to target 12-37 trades/year (50-150 total over 4 years).
Works in bull/bear markets by trading with the 1d trend and using volume to filter false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Donchian channels (20-period) from 6h data
    # Donchian high = max(high, lookback=20)
    # Donchian low = min(low, lookback=20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume spike: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 1d EMA, 20 for Donchian and volume MA)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions
        breakout_long = close[i] > donchian_high[i]
        breakout_short = close[i] < donchian_low[i]
        
        # Re-entry condition: price back inside 20-period range
        price_in_range = (close[i] > donchian_low[i]) and (close[i] < donchian_high[i])
        
        if position == 0:
            # Long: breakout above Donchian high AND close > 1d EMA50 AND volume spike
            if breakout_long and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low AND close < 1d EMA50 AND volume spike
            elif breakout_short and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: breakout below Donchian low OR price re-enters 20-period range
            if breakout_short or price_in_range:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: breakout above Donchian high OR price re-enters 20-period range
            if breakout_long or price_in_range:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyDonchianBreakout_1dTrendFilter_VolumeSpike"
timeframe = "6h"
leverage = 1.0