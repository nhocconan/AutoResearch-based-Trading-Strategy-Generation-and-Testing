#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_TrendFilter_v1
Hypothesis: 4h Donchian(20) breakout with volume spike and 1d EMA50 trend filter.
- Targets 75-200 total trades over 4 years (19-50/year) to minimize fee drag
- Long when price breaks above 20-period high with volume spike and 1d uptrend
- Short when price breaks below 20-period low with volume spike and 1d downtrend
- Uses discrete position sizing (0.25) to reduce churn
- Works in bull (breakouts catch trends) and bear (filters avoid false breakouts in chop)
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
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike (20-period volume average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian/volume, 50 for EMA)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if trend data not ready
        if np.isnan(ema50_1d_aligned[i]):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high_20[i-1]  # Use previous bar's channel
        breakout_down = close[i] < lowest_low_20[i-1]
        
        # 1d trend filter
        trend_up = close[i] > ema50_1d_aligned[i]
        trend_down = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Long: breakout above upper channel + volume spike + 1d uptrend
            if breakout_up and volume_spike[i] and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower channel + volume spike + 1d downtrend
            elif breakout_down and volume_spike[i] and trend_down:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below lower channel OR 1d trend turns down
            if close[i] < lowest_low_20[i] or not trend_up:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above upper channel OR 1d trend turns up
            if close[i] > highest_high_20[i] or not trend_down:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0