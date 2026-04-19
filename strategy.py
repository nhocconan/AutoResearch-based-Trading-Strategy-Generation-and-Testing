#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout + 1w EMA trend + volume confirmation
# Donchian(20) on daily for breakout signals
# 1-week EMA(34) for trend filter
# Daily volume > 1.5x 20-day average for confirmation
# Exit on opposite band touch
# Designed for low frequency (<25 trades/year) to minimize fee drag
# Works in bull/bear via trend filter and breakout logic
name = "1d_Donchian_1wEMA_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1-week EMA(34)
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Donchian channels (20-period) on daily data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-day average
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: break above upper Donchian + above weekly EMA + volume
            if high[i] > highest_high[i] and close[i] > ema_1w_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: break below lower Donchian + below weekly EMA + volume
            elif low[i] < lowest_low[i] and close[i] < ema_1w_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price touches or goes below lower Donchian
            if low[i] <= lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches or goes above upper Donchian
            if high[i] >= highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals