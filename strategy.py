#!/usr/bin/env python3
# 6h_donchian_breakout_12h_trend_volume_v1
# Hypothesis: Donchian(20) breakout on 6h with 12h EMA50 trend filter and volume confirmation captures strong momentum moves while avoiding counter-trend trades.
# Designed for both bull and bear markets by following the higher timeframe trend (12h). Targets 50-150 total trades over 4 years.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_12h_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 6h
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 6h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price below 12h EMA50 OR price touches lower Donchian band
            if (close[i] < ema_50_12h_aligned[i]) or (close[i] <= low_20[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above 12h EMA50 OR price touches upper Donchian band
            if (close[i] > ema_50_12h_aligned[i]) or (close[i] >= high_20[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper Donchian band + volume + price > 12h EMA50
            if (close[i] > high_20[i]) and volume_filter[i] and (close[i] > ema_50_12h_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower Donchian band + volume + price < 12h EMA50
            elif (close[i] < low_20[i]) and volume_filter[i] and (close[i] < ema_50_12h_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals