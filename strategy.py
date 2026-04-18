#!/usr/bin/env python3
"""
4h_1w_Donchian_Breakout_WeeklyTrend_Filter
Hypothesis: Combines 4h Donchian breakouts with weekly trend filter to capture strong trends while avoiding counter-trend trades. Weekly trend ensures alignment with higher timeframe momentum, reducing whipsaws in sideways markets. Designed for ~25-35 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Weekly EMA(34) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume spike: >2.0x 20-period average (more selective)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(60, 34)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        trend = ema_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above Donchian high, above weekly trend, volume spike
            if price > highest_high[i] and price > trend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, below weekly trend, volume spike
            elif price < lowest_low[i] and price < trend and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price breaks below Donchian low or closes below weekly trend
            if price < lowest_low[i] or price < trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price breaks above Donchian high or closes above weekly trend
            if price > highest_high[i] or price > trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_1w_Donchian_Breakout_WeeklyTrend_Filter"
timeframe = "4h"
leverage = 1.0