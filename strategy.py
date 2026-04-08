#!/usr/bin/env python3
"""
12h_1d_donchian_breakout_volume_v1
Hypothesis: 12h Donchian breakout with 1d trend filter and volume confirmation.
- Entry: 12h price breaks above/below 20-period Donchian channel
- Trend filter: 1d EMA(50) direction (long if close > EMA, short if close < EMA)
- Volume: 12h volume > 1.5x 20-period average
- Exit: Opposite Donchian breakout or trend reversal
- Position sizing: 0.25
- Target: 12-30 trades/year (48-120 total over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_donchian_breakout_volume_v1"
timeframe = "12h"
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
    
    # Get 12h data for Donchian and volume
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    vol_12h = df_12h['volume'].values
    
    # 12h Donchian channel (20-period)
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # 12h volume filter
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    volume_filter_12h = vol_12h > (1.5 * vol_ma_12h)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_up = close_1d > ema_50_1d
    trend_1d_down = close_1d < ema_50_1d
    
    # Forward fill trend
    trend_1d_up_series = pd.Series(trend_1d_up)
    trend_1d_down_series = pd.Series(trend_1d_down)
    trend_1d_up_ffilled = trend_1d_up_series.ffill().values
    trend_1d_down_ffilled = trend_1d_down_series.ffill().values
    
    # Align 1d trend to 12h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up_ffilled)
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down_ffilled)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_filter_12h[i]) or
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR 1d trend turns down
            if low_12h[i] < donchian_low[i] or trend_1d_down_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR 1d trend turns up
            if high_12h[i] > donchian_high[i] or trend_1d_up_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high + 1d uptrend + volume
            if high_12h[i] > donchian_high[i] and trend_1d_up_aligned[i] and volume_filter_12h[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low + 1d downtrend + volume
            elif low_12h[i] < donchian_low[i] and trend_1d_down_aligned[i] and volume_filter_12h[i]:
                position = -1
                signals[i] = -0.25
    
    return signals