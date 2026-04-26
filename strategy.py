#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_12hTrend_HTFVolSpike
Hypothesis: 6h Donchian(20) breakout confirmed by 12h EMA50 trend and 1d volume spike (>2.0x 20-period average).
Works in both bull and bear markets because:
- In bull: breaks above upper band capture momentum with trend/volume confirmation
- In bear: breaks below lower band capture short-side momentum with trend/volume confirmation
- Volume spike filter reduces false breakouts in low-participation moves
- Discrete 0.25 position size limits drawdown (max -19.25% in 77% crash)
- Targets ~20-40 trades/year to avoid fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Donchian(20) - using 20-period lookback
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 1d volume spike: current volume > 2.0 * 20-period average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_20_avg_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_20_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_20_avg_1d)
    volume_spike = volume > (2.0 * vol_20_avg_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 20 for Donchian, 50 for 12h EMA, 20 for 1d volume avg
    start_idx = max(lookback, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # Fixed position size
        
        if position == 0:
            # Flat - look for Donchian breakout with trend and volume confirmation
            # Long: break above upper band + 12h EMA50 uptrend + volume spike
            long_entry = (close_val > highest_high[i]) and \
                       (ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]) and \
                       volume_spike[i]
            # Short: break below lower band + 12h EMA50 downtrend + volume spike
            short_entry = (close_val < lowest_low[i]) and \
                        (ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]) and \
                        volume_spike[i]
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price retouches the Donchian middle (mean reversion)
            middle = (highest_high[i] + lowest_low[i]) / 2.0
            if close_val < middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price retouches the Donchian middle (mean reversion)
            middle = (highest_high[i] + lowest_low[i]) / 2.0
            if close_val > middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_Breakout_12hTrend_HTFVolSpike"
timeframe = "6h"
leverage = 1.0