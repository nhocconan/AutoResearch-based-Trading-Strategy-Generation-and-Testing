#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d trend filter and volume confirmation.
# Long when price breaks above 4h Donchian upper (20) + 1d EMA trend up + volume > 1.5x average.
# Short when price breaks below 4h Donchian lower (20) + 1d EMA trend down + volume > 1.5x average.
# Uses volatility-based position sizing (0.25) to manage risk.
# Works in bull (breakouts with trend) and bear (breakouts against trend filtered by 1d EMA).
# Target: 20-50 trades/year to avoid fee drag.

name = "4h_Donchian_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA(34) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_slope = ema_34_1d[1:] - ema_34_1d[:-1]
    ema_34_1d_slope = np.concatenate([[0], ema_34_1d_slope])
    
    # Align 1d EMA slope to 4h
    ema_34_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d_slope)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_34_1d_slope_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > Donchian upper + 1d trend up + volume filter
            if (close[i] > highest_high[i] and 
                ema_34_1d_slope_aligned[i] > 0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short breakout: price < Donchian lower + 1d trend down + volume filter
            elif (close[i] < lowest_low[i] and 
                  ema_34_1d_slope_aligned[i] < 0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian lower or trend fails
            if close[i] < lowest_low[i] or ema_34_1d_slope_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian upper or trend fails
            if close[i] > highest_high[i] or ema_34_1d_slope_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals