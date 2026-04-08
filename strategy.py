#!/usr/bin/env python3
"""
6h_1d_1w_donchian_breakout_volume_v1
Hypothesis: Donchian breakouts with weekly trend filter and volume confirmation.
- Primary: 6h Donchian(20) breakout with volume > 1.5x 20-period average
- Trend filter: 1d close > weekly EMA(50) for longs, < for shorts
- Exit: Opposite Donchian break of 10-period channel
- Position sizing: 0.25 for long, -0.25 for short
Target: 50-150 total trades over 4 years (12-37/year)
Works in bull (breakouts) and bear (short breakdowns) with trend filter preventing counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_donchian_breakout_volume_v1"
timeframe = "6h"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get weekly data for EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1d close values
    close_1d = df_1d['close'].values
    
    # Weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1w_up = close_1w > ema_50_1w
    trend_1w_down = close_1w < ema_50_1w
    
    # Forward fill weekly trend
    trend_1w_up_series = pd.Series(trend_1w_up)
    trend_1w_down_series = pd.Series(trend_1w_down)
    trend_1w_up_ffilled = trend_1w_up_series.ffill().values
    trend_1w_down_ffilled = trend_1w_down_series.ffill().values
    
    # Align weekly trend to 1d (then to 6h)
    trend_1w_up_aligned_1d = align_htf_to_ltf(df_1d, df_1w, trend_1w_up_ffilled)
    trend_1w_down_aligned_1d = align_htf_to_ltf(df_1d, df_1w, trend_1w_down_ffilled)
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1w_up_aligned_1d)
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1w_down_aligned_1d)
    
    # 6h Donchian channels (20-period for entry, 10-period for exit)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i]) or
            np.isnan(volume_filter[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(high_10[i]) or np.isnan(low_10[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below 10-period low OR weekly trend turns down
            if low[i] <= low_10[i] or trend_1w_down_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: price breaks above 10-period high OR weekly trend turns up
            if high[i] >= high_10[i] or trend_1w_up_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: price breaks above 20-period high + weekly uptrend + volume
            if high[i] >= high_20[i] and trend_1w_up_aligned[i] and volume_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below 20-period low + weekly downtrend + volume
            elif low[i] <= low_20[i] and trend_1w_down_aligned[i] and volume_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals