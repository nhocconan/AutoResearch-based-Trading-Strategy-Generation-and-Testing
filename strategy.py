#!/usr/bin/env python3
"""
12h_1d_donchian_breakout_volume_v1
Hypothesis: Donchian channel breakouts on 12h with 1d trend filter and volume confirmation.
- Entry: Price breaks above/below 20-period Donchian channel with volume > 1.5x average
- Trend filter: 1d EMA(50) direction (price above/below EMA)
- Exit: Price returns to middle of Donchian channel (mean reversion)
- Position sizing: 0.30 long, -0.30 short
- Designed to capture trend continuation in bull markets and mean reversion in bear markets
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend
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
    
    # 12h Donchian channel (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_middle = (highest_high + lowest_low) / 2
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(donchian_middle[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price returns to Donchian middle OR trend turns down
            if (close[i] <= donchian_middle[i]) or trend_1d_down_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30  # Position size
                
        elif position == -1:  # Short position
            # Exit: Price returns to Donchian middle OR trend turns up
            if (close[i] >= donchian_middle[i]) or trend_1d_up_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30  # Position size
        else:  # Flat, look for entry
            # Long entry: Price breaks above Donchian upper + 1d uptrend + volume
            if (close[i] > highest_high[i]) and trend_1d_up_aligned[i] and volume_filter[i]:
                # Confirm breakout (close above previous high)
                if i > start_idx and close[i-1] <= highest_high[i-1]:
                    position = 1
                    signals[i] = 0.30
            # Short entry: Price breaks below Donchian lower + 1d downtrend + volume
            elif (close[i] < lowest_low[i]) and trend_1d_down_aligned[i] and volume_filter[i]:
                # Confirm breakdown (close below previous low)
                if i > start_idx and close[i-1] >= lowest_low[i-1]:
                    position = -1
                    signals[i] = -0.30
    
    return signals