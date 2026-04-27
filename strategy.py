#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dVolume_1wTrend_v1
Hypothesis: Donchian(20) breakouts on 4h capture momentum bursts. 
Filter by 1d volume surge (>2x 20-period average) to avoid false breakouts.
Use 1w EMA50 trend filter to ensure alignment with higher-timeframe direction.
Works in bull (breakouts with volume) and bear (avoids counter-trend trades via 1w trend).
Target: ~25-40 trades/year to minimize fee drag on 4h.
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
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume 20-period average
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_surge_1d = df_1d['volume'].values > (vol_ma_1d * 2.0)
    vol_surge_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_surge_1d)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 trend
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Donchian(20) channels on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for Donchian, 1d volume, 1w trend
    start_idx = max(20, 30, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or \
           np.isnan(vol_surge_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper, 1d volume surge, price > 1w EMA50 (uptrend)
            if close[i] > highest_high[i] and vol_surge_1d_aligned[i] and close[i] > ema50_1w_aligned[i]:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian lower, 1d volume surge, price < 1w EMA50 (downtrend)
            elif close[i] < lowest_low[i] and vol_surge_1d_aligned[i] and close[i] < ema50_1w_aligned[i]:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian middle or trend weakens
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2.0
            if close[i] < donchian_mid or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above Donchian middle or trend weakens
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2.0
            if close[i] > donchian_mid or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_1dVolume_1wTrend_v1"
timeframe = "4h"
leverage = 1.0