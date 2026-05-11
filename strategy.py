# 6h_Donchian20_1dTrend_VolumeBreakout
# Hypothesis: Donchian breakouts on 6h timeframe with 1d trend filter and volume confirmation
# captures breakout momentum while avoiding false signals in choppy markets.
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue).
# Volume filter ensures only significant breakouts are traded, reducing whipsaws.
# Target: 12-37 trades/year per symbol.

#!/usr/bin/env python3
name = "6h_Donchian20_1dTrend_VolumeBreakout"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Donchian channels (20-period) on 6h data
    # Use rolling window with min_periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection (20-period average)
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)  # Require 2x average volume for breakout
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above Donchian high + above 1d EMA50 + volume spike
            if close[i] > donchian_high[i] and close[i] > ema50_1d_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + below 1d EMA50 + volume spike
            elif close[i] < donchian_low[i] and close[i] < ema50_1d_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals