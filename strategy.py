#!/usr/bin/env python3
# 4h_donchian_breakout_1d_trend_v2
# Hypothesis: 4-hour Donchian channel breakout (20-period) in direction of daily trend (price above/below daily EMA50).
# Adds volume confirmation (volume > 1.5x 20-period average) to ensure participation.
# Designed to work in bull markets (breakouts) and bear markets (breakdowns).
# Target: 25-40 trades/year with 0.25 position size to minimize fee drag.

name = "4h_donchian_breakout_1d_trend_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i - 19:i + 1])
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    close_daily = df_daily['close'].values
    
    # Daily EMA (50-period) for trend filter
    ema_period = 50
    ema_daily = np.full_like(close_daily, np.nan)
    for i in range(ema_period - 1, len(close_daily)):
        if i == ema_period - 1:
            ema_daily[i] = np.mean(close_daily[:ema_period])
        else:
            ema_daily[i] = (close_daily[i] * 2 + ema_daily[i-1] * (ema_period - 1)) / (ema_period + 1)
    
    # Align daily EMA to 4h timeframe
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = max(lookback, ema_period) + 5
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i]) or volume[i] == 0 or 
            np.isnan(ema_daily_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        # Higher timeframe trend filter: price above/below daily EMA
        uptrend_htf = close[i] > ema_daily_aligned[i]
        downtrend_htf = close[i] < ema_daily_aligned[i]
        
        if position == 1:  # Long position
            # Exit if trend reverses, volume fails, or price breaks below Donchian low
            if not uptrend_htf or not volume_filter or close[i] < lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if trend reverses, volume fails, or price breaks above Donchian high
            if not downtrend_htf or not volume_filter or close[i] > highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high, volume breakout, daily uptrend
            if close[i] > highest_high[i] and volume_filter and uptrend_htf:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low, volume breakout, daily downtrend
            elif close[i] < lowest_low[i] and volume_filter and downtrend_htf:
                position = -1
                signals[i] = -0.25
    
    return signals