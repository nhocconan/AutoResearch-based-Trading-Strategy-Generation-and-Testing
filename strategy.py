#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d trend filter and volume confirmation.
# Uses Donchian breakout for entry, 1d EMA50 for trend direction, and volume > 1.5x 20-period average for confirmation.
# Long when price breaks above Donchian upper band, 1d trend up, and volume confirmation.
# Short when price breaks below Donchian lower band, 1d trend down, and volume confirmation.
# Exit when price crosses the Donchian midline (20-period average of high/low).
# Designed for 6h timeframe to capture medium-term trends with filtered breakouts.
# Works in both bull (follow 1d trend) and bear (follow 1d trend) markets.

name = "6h_Donchian_Breakout_1dTrend_Volume"
timeframe = "6h"
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
    
    # Donchian channels (20-period)
    highest_high = np.zeros_like(high)
    lowest_low = np.zeros_like(low)
    highest_high[:] = np.nan
    lowest_low[:] = np.nan
    
    for i in range(19, len(high)):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    # Donchian breakout signals
    breakout_up = (high > highest_high) & ~np.isnan(highest_high)
    breakout_down = (low < lowest_low) & ~np.isnan(lowest_low)
    
    # Donchian midline for exit (average of highest high and lowest low)
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # 1d EMA(50) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_up = ema_50_1d[1:] > ema_50_1d[:-1]
    trend_1d_up = np.concatenate([[False], trend_1d_up])
    
    # Align 1d trend to 6h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.zeros_like(volume)
    vol_ma[:] = np.nan
    for i in range(19, len(volume)):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for breakout entries with trend and volume confirmation
            if breakout_up[i] and trend_1d_up_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            elif breakout_down[i] and not trend_1d_up_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below Donchian midline
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above Donchian midline
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals