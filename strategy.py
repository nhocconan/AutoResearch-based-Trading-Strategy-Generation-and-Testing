#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter and volume confirmation
# Donchian breakouts capture momentum bursts; 1d EMA50 filters direction; volume confirms strength.
# Works in bull (breakouts above) and bear (breakdowns below). Target: 25-40 trades/year per symbol.

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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Donchian channels (20-period) on 4h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long: price breaks above upper Donchian + uptrend + volume
        if (close[i] > high_20[i-1] and  # Breakout above prior 20-period high
            close[i] > ema50_1d_aligned[i] and  # Uptrend filter
            volume_filter[i]):
            signals[i] = 0.30
            position = 1
        # Short: price breaks below lower Donchian + downtrend + volume
        elif (close[i] < low_20[i-1] and  # Breakdown below prior 20-period low
              close[i] < ema50_1d_aligned[i] and  # Downtrend filter
              volume_filter[i]):
            signals[i] = -0.30
            position = -1
        else:
            # Hold current position
            signals[i] = 0.30 if position == 1 else (-0.30 if position == -1 else 0.0)
    
    return signals

name = "4h_Donchian20_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0