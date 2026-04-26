#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1wTrend_VolumeSpike
Hypothesis: 6h Donchian(20) breakout with 1w trend filter and volume confirmation (2.0x). 
Designed to capture medium-term trends with tight entries (target 50-150 trades over 4 years) 
by requiring confluence of price breakout, weekly trend alignment, and elevated volume. 
Uses discrete position sizing (0.25) to minimize fee churn. Works in bull/bear via 1w trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    # Trend: 1 = uptrend (close > EMA50), -1 = downtrend (close < EMA50), 0 = invalid
    trend_1w = np.where(ema_50_1w_aligned > 0, 
                        np.where(close > ema_50_1w_aligned, 1, -1), 
                        0)
    
    # Calculate Donchian channels (20-period) on 6h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 2.0 * volume_ma(30) for strong confirmation
    volume_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for EMA, 20 for Donchian, 30 for volume MA)
    start_idx = max(50, 20, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or np.isnan(volume_ma[i]) or
            np.isnan(trend_1w[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Donchian breakout conditions with 1w trend and volume confirmation
        if position == 0:
            # Long: Price breaks above Donchian upper band AND 1w uptrend AND volume spike (2.0x)
            if close[i] > highest_high[i] and trend_1w[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower band AND 1w downtrend AND volume spike (2.0x)
            elif close[i] < lowest_low[i] and trend_1w[i] == -1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Donchian lower band OR 1w trend turns down
            if close[i] < lowest_low[i] or trend_1w[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Donchian upper band OR 1w trend turns up
            if close[i] > highest_high[i] or trend_1w[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_1wTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0