# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
4h_1d_donchian_breakout_volume_v1
Hypothesis: 4h Donchian channel breakouts with 1d trend filter and volume confirmation work in both bull and bear markets.
- Trend: 1d close above/below EMA(50) determines long/short bias
- Entry: 4h price breaks Donchian(20) high/low in direction of 1d trend with volume > 1.5x 20-period average
- Exit: Opposite Donchian(20) touch or trend reversal
- Volume: Require 4h volume > 1.5x 20-period average to avoid false breakouts
Target: 20-50 trades/year (80-200 total over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up = close_1d > ema_50
    trend_down = close_1d < ema_50
    
    # Forward fill trend
    trend_up_series = pd.Series(trend_up)
    trend_down_series = pd.Series(trend_down)
    trend_up_ffilled = trend_up_series.ffill().values
    trend_down_ffilled = trend_down_series.ffill().values
    
    # Align 1d trend to 4h
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up_ffilled)
    trend_down_aligned = align_htf_to_ltf(prices, df_1d, trend_down_ffilled)
    
    # Donchian channel (20-period) on 4h
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 4h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i]) or 
            np.isnan(high_max[i]) or np.isnan(low_min[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price touches Donchian low or trend turns down
            if low[i] <= low_min[i] or trend_down_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: price touches Donchian high or trend turns up
            if high[i] >= high_max[i] or trend_up_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high with 1d uptrend and volume
            if high[i] > high_max[i] and trend_up_aligned[i] and volume_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low with 1d downtrend and volume
            elif low[i] < low_min[i] and trend_down_aligned[i] and volume_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals